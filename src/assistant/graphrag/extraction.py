"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

# Ответ модели на чанк руководства содержит до трёх десятков сущностей: 1024 токенов может не хватить.
MAX_TOKENS = 2048

# Типы с пояснениями: без них модель путает Component и ProductLine.
_TYPES = (
    "Типы сущностей: Model — код модели; Part — номер детали; ErrorCode — код ошибки; "
    "Procedure — номер регламента; ProductLine — название линейки; Component — название узла; "
    "Supplier — название поставщика."
)

_DIRECTIONS = "\n".join(f"{kind}: {source} -> {target}" for kind, (source, target) in RELATION_SIGNATURES.items())

_EXAMPLE = (
    "Пример ответа на текст «В модели AR-210 установлена деталь P-1042 (пусковое реле), "
    "поставщик «ХолодКомплект»»:\n"
    '{"entities": [{"type": "Model", "name": "AR-210"}, {"type": "Part", "name": "P-1042"},\n'
    '{"type": "Component", "name": "пусковое реле"}, {"type": "Supplier", "name": "ХолодКомплект"}],\n'
    '"relations": [{"subject_type": "Model", "subject": "AR-210", "type": "HAS_PART", "object_type": "Part",\n'
    '"object": "P-1042"}, {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": "Component",\n'
    '"object": "пусковое реле"}, {"subject_type": "Part", "subject": "P-1042", "type": "SUPPLIED_BY",\n'
    '"object_type": "Supplier", "object": "ХолодКомплект"}]}'
)

_RULES = (
    "Правила:\n"
    "- Возьми все сущности и все связи, которые прямо названы в тексте. Пропуск названной связи — ошибка.\n"
    "- Коды пиши заглавными латинскими буквами с дефисом: AR-210, P-1042B, E-17, SP-104.\n"
    "- Номера бюллетеней и отчётов (SB-2025-03, R-2025-017) — имена документов, а не сущности.\n"
    "- Строка состава узлов «узел – деталь P-1010, поставщик «КомпрессорМаш» даёт три связи: "
    "Model HAS_PART Part, Part IS_A Component, Part SUPPLIED_BY Supplier.\n"
    "- Строка кодов ошибок «E-17 – ..., отказ узла «пусковое реле»» даёт ErrorCode INDICATES Component, "
    "а фраза «устраняется по регламенту SP-104» — ErrorCode FIXED_BY Procedure.\n"
    "- В бюллетене «Деталь P-1042 (пусковое реле) снята с производства. Вместо неё P-1042B» нужны связи "
    "Part P-1042 IS_A Component пусковое реле, Part P-1042 REPLACED_BY Part P-1042B, "
    "Part P-1042B SUPPLIED_BY Supplier.\n"
    "- Фраза отчёта «отказ узла: пусковое реле» даёт Model FAILED Component модели из первой строки.\n"
)

_RETRY_MESSAGE = {
    "role": "user",
    "content": "Предыдущий ответ не разобрался по схеме. Верни только корректный JSON с полями entities и relations.",
}

# Маленькая модель теряет связи из состава узлов, поэтому пропуск проверяется кодом и исправляется отдельно.
_MISSING_ROWS = (
    "В ответе не хватает связей, которые прямо названы в тексте:\n{rows}\n"
    "Верни полный JSON заново, добавив эти связи и сохранив все остальные."
)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка на нечитаемый ответ, затем
    пустой результат. Если разобранный ответ потерял связи, прямо названные в строках состава узлов,
    делается один корректирующий запрос, и его связи добавляются к уже полученным.
    """
    names = vocabulary()
    known_codes = set(find_codes(chunk.text))
    request = _messages(chunk, names)
    reply = None
    payload = None
    for attempt in range(2):
        reply = llm.chat(request, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_TOKENS)
        payload = _parse_payload(reply.content)
        if payload is not None:
            break
        if attempt == 0:
            # Второй запрос отличается от первого, поэтому кеш не вернёт тот же нечитаемый ответ.
            request = [*request, _RETRY_MESSAGE]
    if payload is None or reply is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    payload = _complete_listed_rows(chunk, llm, request, reply, payload, names)
    return _collect(chunk, payload, names, known_codes)


def _messages(chunk: Chunk, names: dict[str, list[str]]) -> list[dict]:
    system = (
        "Ты извлекаешь факты из документов сервисной службы компании «ПолярТех».\n"
        "Верни ровно один JSON-объект с полями entities и relations, без пояснений и без markdown.\n\n"
        f"{_TYPES}\n\n"
        f"Связи и направление:\n{_DIRECTIONS}\n\n"
        f"{_RULES}\n"
        f"{_EXAMPLE}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Чанк документа {chunk.doc_id}:\n\n{chunk.text}\n\n{_hints(chunk, names)}"},
    ]


def _hints(chunk: Chunk, names: dict[str, list[str]]) -> str:
    """Подсказка о модели документа, посчитанная кодом по тексту чанка.

    Подсказка не добавляет фактов: это тот же код модели, что и в тексте, только найденный без модели.
    Перечислять в подсказке все коды чанка нельзя: тогда модель считает состав узлов уже покрытым
    сущностями и перестаёт выдавать `HAS_PART`.
    """
    codes = find_codes(chunk.text)
    models = [code for code in codes if code_type(code) == "Model"]
    if not models:
        return ""
    if len(models) > 1:
        return f"Подсказка: в чанке упомянуты модели {', '.join(models)}. Новые факты не добавляй."
    return (
        f"Подсказка: документ описывает модель {models[0]}; "
        f"в связях HAS_PART и FAILED начало связи — Model:{models[0]}. Новые факты не добавляй."
    )


def _parse_payload(text: str) -> dict | None:
    """Разбирает ответ модели. Возвращает None, если ответ не соответствует схеме."""
    cleaned = _strip_fences(text)
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if not isinstance(data.get("entities"), list) or not isinstance(data.get("relations"), list):
        return None
    for item in data["entities"]:
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in ("type", "name")):
            return None
    for item in data["relations"]:
        keys = ("subject_type", "subject", "type", "object_type", "object")
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) for key in keys):
            return None
    return data


def _strip_fences(text: str) -> str:
    """Снимает markdown-обёртку, которой некоторые модели окружают JSON."""
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    if "\n" in cleaned:
        cleaned = cleaned.split("\n", 1)[1]
    cleaned = cleaned.rstrip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _listed_rows(chunk: Chunk, names: dict[str, list[str]]) -> list[tuple[str, str, str, str, str]]:
    """Связи состава узлов, найденные кодом в строках вида «узел – деталь P-1010, поставщик «...»».

    Одна строка даёт до трёх связей: `Model HAS_PART Part`, `Part IS_A Component`,
    `Part SUPPLIED_BY Supplier`. Разбор идёт по тексту чанка, поэтому не зависит от ответа модели.
    """
    models = [code for code in find_codes(chunk.text) if code_type(code) == "Model"]
    if len(models) != 1:
        return []
    rows: list[tuple[str, str, str, str, str]] = []
    for line in chunk.text.splitlines():
        if not line.lstrip().startswith("- ") or "детал" not in line.lower():
            continue
        parts = [code for code in find_codes(line) if code_type(code) == "Part"]
        if not parts:
            continue
        part = parts[0]
        rows.append(("Model", models[0], "HAS_PART", "Part", part))
        component = canonical_name("Component", _before_dash(line), names)
        if component:
            rows.append(("Part", part, "IS_A", "Component", component))
        supplier = canonical_name("Supplier", line, names)
        if supplier:
            rows.append(("Part", part, "SUPPLIED_BY", "Supplier", supplier))
    return rows


def _before_dash(line: str) -> str:
    """Часть строки до тире: в составе узлов там стоит название узла."""
    for dash in ("–", "—", " - "):
        if dash in line:
            return line.split(dash, 1)[0].lstrip("- ").strip()
    return ""


def _triples(payload: dict, names: dict[str, list[str]]) -> set[tuple[str, str, str, str, str]]:
    """Связи ответа в каноническом виде, пригодные для сравнения с ожидаемыми."""
    found = set()
    for item in payload["relations"]:
        subject = canonical_name(item["subject_type"], item["subject"], names)
        obj = canonical_name(item["object_type"], item["object"], names)
        if subject is None or obj is None:
            continue
        found.add((item["subject_type"], subject, item["type"], item["object_type"], obj))
    return found


def _complete_listed_rows(
    chunk: Chunk,
    llm: LLMClient,
    request: list[dict],
    reply,
    payload: dict,
    names: dict[str, list[str]],
) -> dict:
    """Один корректирующий запрос, если модель потеряла связи из состава узлов.

    Проверка выполняется кодом: строки состава узлов разбираются из текста чанка, а не из ответа модели.
    Ответ модели нестабилен на одинаковых руководствах: в части документов она опускает `HAS_PART`,
    иногда вместе с `IS_A`. Связи второго ответа добавляются к первым, поэтому найденное не теряется.
    """
    expected = _listed_rows(chunk, names)
    if not expected:
        return payload
    missing = sorted(set(expected) - _triples(payload, names))
    if not missing:
        return payload
    rows = "\n".join(
        f"- {subject_type}:{subject} {kind} {object_type}:{obj}"
        for subject_type, subject, kind, object_type, obj in missing
    )
    follow_up = [
        *request,
        reply.as_message(),
        {"role": "user", "content": _MISSING_ROWS.format(rows=rows)},
    ]
    second = llm.chat(follow_up, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_TOKENS)
    extra = _parse_payload(second.content)
    if extra is None:
        return payload
    return _merge(payload, extra)


def _merge(base: dict, extra: dict) -> dict:
    """Объединяет два ответа модели, убирая повторы по сущности и по связи."""
    entities = list(base["entities"])
    seen_entities = {(item["type"], item["name"]) for item in entities}
    for item in extra["entities"]:
        key = (item["type"], item["name"])
        if key not in seen_entities:
            seen_entities.add(key)
            entities.append(item)

    relations = list(base["relations"])
    seen_relations = {
        (item["subject_type"], item["subject"], item["type"], item["object_type"], item["object"])
        for item in relations
    }
    for item in extra["relations"]:
        key = (item["subject_type"], item["subject"], item["type"], item["object_type"], item["object"])
        if key not in seen_relations:
            seen_relations.add(key)
            relations.append(item)
    return {"entities": entities, "relations": relations}


def _collect(
    chunk: Chunk, payload: dict, names: dict[str, list[str]], known_codes: set[str]
) -> ExtractionResult:
    """Оставляет только подтверждённые сущности и связи и считает отброшенное."""
    result = ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
    kept_entities: dict[str, Entity] = {}
    kept_relations: dict[tuple[str, str, str], Relation] = {}
    dropped = 0

    for raw in payload["entities"]:
        entity = _confirmed(raw["type"], raw["name"], names, known_codes)
        if entity is None:
            dropped += 1
            continue
        kept_entities.setdefault(entity.node_id, entity)

    for raw in payload["relations"]:
        kind = raw["type"]
        subject = _confirmed(raw["subject_type"], raw["subject"], names, known_codes)
        obj = _confirmed(raw["object_type"], raw["object"], names, known_codes)
        if kind not in RELATION_SIGNATURES or subject is None or obj is None:
            dropped += 1
            continue
        relation = Relation(subject, kind, obj)
        if not relation.matches_schema():
            dropped += 1
            continue
        if _self_replacement(relation):
            # На бюллетенях модель пишет «P-1042 заменена на P-1042» и теряет новую деталь.
            # Второй конец берётся из той же фразы текста, где названа замена.
            repaired = _replacement_target(chunk, relation.subject)
            if repaired is None:
                dropped += 1
                continue
            relation = Relation(relation.subject, "REPLACED_BY", repaired)
            subject, obj = relation.subject, relation.object
        kept_relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)
        # Концы выжившей связи — тоже сущности чанка, даже если модель забыла их в списке entities.
        kept_entities.setdefault(subject.node_id, subject)
        kept_entities.setdefault(obj.node_id, obj)

    result.entities = list(kept_entities.values())
    result.relations = list(kept_relations.values())
    result.dropped = dropped
    return result


def _self_replacement(relation: Relation) -> bool:
    """Замена детали самой собой бессмысленна и рвёт цепочки `REPLACED_BY`."""
    return relation.type == "REPLACED_BY" and relation.subject.node_id == relation.object.node_id


def _replacement_target(chunk: Chunk, subject: Entity) -> Entity | None:
    """Новая деталь из фразы о замене: «Вместо неё устанавливается деталь P-1042B».

    Модель в бюллетенях регулярно повторяет старую деталь в обоих концах `REPLACED_BY`.
    Код находит в тексте предложение про снятие с производства и берёт из него деталь,
    отличную от заменяемой. Если такой детали нет, связь действительно отбрасывается.
    """
    for sentence in chunk.text.replace("\n", " ").split(". "):
        if "вместо" not in sentence.lower():
            continue
        for code in find_codes(sentence):
            if code_type(code) == "Part" and code != subject.name:
                return Entity("Part", code)
    return None


def _confirmed(
    entity_type: str, raw_name: str, names: dict[str, list[str]], known_codes: set[str]
) -> Entity | None:
    """Каноническая сущность или None, если её нет в словаре либо кода нет в тексте чанка."""
    name = canonical_name(entity_type, raw_name, names)
    if name is None:
        return None
    if entity_type in CODE_TYPES and name not in known_codes:
        return None
    return Entity(entity_type, name)
