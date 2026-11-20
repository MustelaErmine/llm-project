"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

# Отказ по FR-10: данных в базе знаний нет.
REFUSAL = "В базе знаний не найдено."

# Пути между опорными сущностями ищутся не длиннее трёх связей (FR-4).
PATH_LENGTH = 3

# Окрестность одной опорной сущности — до двух связей (FR-4).
NEIGHBOURHOOD = 2

# Сколько символов списка фактов получает модель (FR-4).
FACTS_LIMIT = 8000

# Замена может быть цепочкой: P-2054 -> P-2054B -> P-2054C. Ограничение защищает от циклов.
REPLACEMENT_LIMIT = 10

LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос только по пронумерованным фактам ниже.\n"
    "Коды моделей, деталей, ошибок и регламентов пиши точно как в фактах.\n"
    "Как читать связи:\n"
    "- BELONGS_TO: Model -BELONGS_TO-> ProductLine — модель входит в линейку.\n"
    "- HAS_PART: Model -HAS_PART-> Part — в модели стоит эта деталь.\n"
    "- IS_A: Part -IS_A-> Component — деталь является этим узлом.\n"
    "- INDICATES: ErrorCode -INDICATES-> Component — ошибка указывает на отказ узла.\n"
    "- FIXED_BY: ErrorCode -FIXED_BY-> Procedure — сбой устраняют по этому регламенту (SP-…).\n"
    "- FAILED: Model -FAILED-> Component — в модели отказывал этот узел.\n"
    "- SUPPLIED_BY: Part -SUPPLIED_BY-> Supplier — деталь поставляет этот поставщик.\n"
    "- REPLACED_BY: Part -REPLACED_BY-> Part — деталь заменена новой.\n"
    "  Цепочка замен может быть длиной в две и более замены: P-2054 -> P-2054B -> P-2054C.\n"
    "  Если вопрос спрашивает, какая деталь сейчас нужна, переходи по REPLACED_BY до конца\n"
    "  и называй самую последнюю деталь цепочки.\n"
    "Цен и остатков на складе в фактах нет: не называй их, даже если о них спрашивают.\n"
    "Называй только то, что нужно заказать.\n"
    "Отвечай одной короткой фразой: только нужный код или название, без разбора фактов.\n"
    f"Если ответа в фактах нет, напиши: «{REFUSAL.lower()}»."
)

GLOBAL_SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Ниже сводки сообществ графа знаний.\n"
    "Ответь на вопрос по сводке. Если сводка не помогает, поставь оценку 0.\n"
    'Верни JSON: {"answer": "частичный ответ", "score": 0-10}.'
)

REDUCE_SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Тебе даны частичные ответы с оценкой полезности 0-10.\n"
    "Ответь одной короткой фразой по самому полезному ответу: у него наибольшая оценка.\n"
    "Если наибольшую оценку делят несколько ответов, объедини их факты в одну фразу.\n"
    "Не рассуждай, не проверяй полноту и не пиши того, чего нет в ответах. Отказываться нельзя."
)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    anchors, refused = _anchors(question, kb)
    if refused:
        return Answer(REFUSAL, refused=True)
    if not anchors:
        return Answer(REFUSAL, refused=True)
    edges = _facts_for_anchors(anchors, kb.graph)
    if not edges:
        return Answer(REFUSAL, refused=True)
    facts = _fact_lines(edges, kb.graph)
    sources = _sources(edges, kb.graph)
    reply = llm.chat(
        [
            {"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"Факты:\n{facts}\n\nВопрос: {question}"},
        ]
    )
    refused = "не найдено" in reply.content.lower()
    return Answer(
        reply.content,
        sources=sources,
        facts=[line.split(". ", 1)[-1] for line in facts.splitlines()],
        refused=refused,
    )


def _anchors(question: str, kb: KnowledgeBase) -> tuple[list[str], bool]:
    """Опорные сущности вопроса и признак отказа: код вопроса отсутствует в графе."""
    anchors: list[str] = []
    for code in find_codes(question):
        node_id = f"{code_type(code)}:{code}"
        if node_id not in kb.graph:
            return [], True
        if node_id not in anchors:
            anchors.append(node_id)
    for kind, name in find_names(question, kb.vocabulary):
        node_id = f"{kind}:{name}"
        if node_id in kb.graph and node_id not in anchors:
            anchors.append(node_id)
    return anchors, False


def _facts_for_anchors(anchors: list[str], graph: nx.DiGraph) -> list[tuple[str, str, str]]:
    """Рёбра фактов: пути между опорными сущностями, окрестность одной, цепочки замен.

    Порядок фактов важен: список обрезается по длине, а нужная связь часто лежит на втором шаге
    от опорной сущности. Поэтому сначала идут факты, где участвует сама опорная сущность,
    затем — связи, которые отвечают на вопрос (`INDICATES`, `FIXED_BY`, `REPLACED_BY`),
    и только потом массовые `HAS_PART` и `SUPPLIED_BY` состава узлов.
    """
    if len(anchors) >= 2:
        nodes = _nodes_on_paths(anchors, graph)
    else:
        nodes = _neighbourhood(anchors[0], graph)
    edges = _edges_inside(nodes, graph)
    edges |= _replacement_chains(nodes, graph)
    distance = _anchor_distance(anchors, nodes, graph)
    return sorted(edges, key=lambda item: _rank(item, distance, graph))


# Чем меньше число, тем раньше связь попадает в список фактов.
_RELATION_PRIORITY = {
    "INDICATES": 0,
    "FIXED_BY": 0,
    "REPLACED_BY": 1,
    "FAILED": 2,
    "BELONGS_TO": 3,
    "IS_A": 4,
    "SUPPLIED_BY": 5,
    "HAS_PART": 6,
}


def _anchor_distance(anchors: list[str], nodes: set[str], graph: nx.DiGraph) -> dict[str, int]:
    """Кратчайшее расстояние от ближайшей опорной сущности до каждого отобранного узла."""
    undirected = graph.to_undirected(as_view=True)
    distance: dict[str, int] = {}
    for anchor in anchors:
        for node, length in nx.single_source_shortest_path_length(undirected, anchor, cutoff=PATH_LENGTH).items():
            if node in nodes and length < distance.get(node, PATH_LENGTH + 1):
                distance[node] = length
    return distance


def _rank(
    edge: tuple[str, str, str], distance: dict[str, int], graph: nx.DiGraph
) -> tuple[int, int, str, str]:
    """Ключ сортировки факта: близость к опорной сущности, важность связи, затем имена."""
    source, kind, target = edge
    near = min(distance.get(source, PATH_LENGTH), distance.get(target, PATH_LENGTH))
    return (near, _RELATION_PRIORITY.get(kind, len(_RELATION_PRIORITY)), graph.nodes[source]["name"], target)



def _nodes_on_paths(anchors: list[str], graph: nx.DiGraph) -> set[str]:
    """Узлы всех путей длиной до трёх связей между опорными сущностями, направление игнорируется."""
    undirected = graph.to_undirected(as_view=True)
    nodes: set[str] = set(anchors)
    for index, start in enumerate(anchors):
        for target in anchors[index + 1 :]:
            if start == target or not nx.has_path(undirected, start, target):
                continue
            for path in nx.all_simple_paths(undirected, start, target, cutoff=PATH_LENGTH):
                nodes.update(path)
    return nodes


def _neighbourhood(anchor: str, graph: nx.DiGraph) -> set[str]:
    """Окрестность одной опорной сущности на глубину до двух связей в любую сторону."""
    undirected = graph.to_undirected(as_view=True)
    return set(nx.single_source_shortest_path_length(undirected, anchor, cutoff=NEIGHBOURHOOD))


def _edges_inside(nodes: set[str], graph: nx.DiGraph) -> set[tuple[str, str, str]]:
    """Рёбра графа, оба конца которых попали в отобранные узлы."""
    edges = set()
    for source, target in graph.edges:
        if source in nodes and target in nodes:
            edges.add((source, graph.edges[source, target]["type"], target))
    return edges


def _replacement_chains(nodes: set[str], graph: nx.DiGraph) -> set[tuple[str, str, str]]:
    """Цепочки `REPLACED_BY` от деталей на путях и связи последней детали цепочки."""
    edges: set[tuple[str, str, str]] = set()
    for node in list(nodes):
        if graph.nodes[node].get("type") != "Part":
            continue
        current = node
        for _ in range(REPLACEMENT_LIMIT):
            replacements = [
                target
                for target in graph.successors(current)
                if graph.edges[current, target]["type"] == "REPLACED_BY"
            ]
            if not replacements:
                break
            target = sorted(replacements)[0]
            edges.add((current, "REPLACED_BY", target))
            current = target
        # Последняя деталь цепочки отвечает на вопрос «что заказывать», поэтому её связи нужны.
        if current != node:
            for _, target, attrs in graph.out_edges(current, data=True):
                if attrs["type"] != "REPLACED_BY":
                    edges.add((current, attrs["type"], target))
    return edges


def _fact_lines(edges: list[tuple[str, str, str]], graph: nx.DiGraph) -> str:
    """Пронумерованный список фактов вида `E-17 -INDICATES-> пусковое реле` не длиннее 8000 символов."""
    lines: list[str] = []
    length = 0
    for number, (source, kind, target) in enumerate(edges, start=1):
        line = (
            f"{number}. {graph.nodes[source]['name']} -{kind}-> {graph.nodes[target]['name']}"
        )
        if lines and length + len(line) + 1 > FACTS_LIMIT:
            break
        lines.append(line)
        length += len(line) + 1
    return "\n".join(lines)


def _sources(edges: set[tuple[str, str, str]], graph: nx.DiGraph) -> list[str]:
    """Документы-источники фактов: только идентификаторы документов, без номеров чанков.

    Берутся документы самих рёбер, а не узлов: узел упоминается во многих документах,
    и список источников раздувается до десятков файлов, среди которых нет нужного.
    """
    documents: list[str] = []
    for source, _, target in sorted(edges):
        for chunk_id in graph.edges[source, target].get("sources", []):
            document = chunk_id.split("#")[0]
            if document not in documents:
                documents.append(document)
    return documents


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials: list[tuple[int, str]] = []
    sources: list[str] = []
    # Тот же отказ без вызова модели, что и в local: кода вопроса нет в графе знаний.
    if any(f"{code_type(code)}:{code}" not in kb.graph for code in find_codes(question)):
        return Answer(REFUSAL, refused=True)
    for community in kb.communities:
        summary = f"Сводка сообщества {community.community_id}:\n{community.summary}"
        reply = llm.chat(
            [
                {"role": "system", "content": GLOBAL_SYSTEM},
                {"role": "user", "content": f"{summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        partial = _partial_answer(reply.content)
        if partial is None:
            continue
        answer, score = partial
        if score <= 0 or not answer.strip():
            continue
        partials.append((score, answer.strip()))
        for document in _community_sources(community, kb.graph):
            if document not in sources:
                sources.append(document)
    if not partials:
        return Answer(REFUSAL, refused=True)
    # Отказные пересказы сводки («в сводке нет...») оцениваются высоко просто за честность,
    # но ответа не содержат: на шаг reduce идут только частичные ответы с самим ответом.
    useful = [(score, text) for score, text in partials if not _is_negative(text)]
    if not useful:
        return Answer(REFUSAL, sources=sources, refused=True)
    # Модель на шаге reduce сбивается на порядок списка, а не на оценки. Поэтому на объединение
    # уходят только ответы с наибольшей оценкой: остальные сами признали себя менее полезными.
    # Ответы с равной наибольшей оценкой остаются все — их факты и нужно объединить.
    best = max(score for score, _ in useful)
    useful = [(score, text) for score, text in useful if score == best]
    useful.sort(key=lambda item: item[1])
    listed = "\n".join(f"{number}. [оценка {score}] {text}" for number, (score, text) in enumerate(useful, start=1))
    reply = llm.chat(
        [
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": f"Частичные ответы:\n{listed}\n\nВопрос: {question}"},
        ]
    )
    refused = "не найдено" in reply.content.lower()
    return Answer(reply.content, sources=sources, facts=[text for _, text in useful], refused=refused)


def _is_negative(text: str) -> bool:
    """Частичный ответ не содержит ответа, а лишь сообщает, что в сводке его нет."""
    lowered = text.lower()
    return "не содержится" in lowered or "не указано" in lowered or "не указаны" in lowered or "отсутствует" in lowered


def _partial_answer(text: str) -> tuple[str, int] | None:
    """Частичный ответ шага map. Нечитаемый ответ пропускается, как требует FR-5."""
    try:
        data = json.loads(text.strip())
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return None
    score = data.get("score")
    if not isinstance(score, int) or isinstance(score, bool):
        return None
    return data["answer"], score


def _community_sources(community, graph: nx.DiGraph) -> list[str]:
    """Документы, на которых построены узлы сообщества."""
    documents: list[str] = []
    for node in community.nodes:
        if node not in graph:
            continue
        for chunk_id in graph.nodes[node].get("sources", []):
            document = chunk_id.split("#")[0]
            if document not in documents:
                documents.append(document)
    return documents
