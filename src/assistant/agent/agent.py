"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import NOT_FOUND, TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

# Отказ по FR-10.
REFUSAL = "В базе знаний не найдено."

# Сколько раз повторять вызов ERP при `ErpUnavailable`: первая попытка и ещё две (FR-8).
ERP_ATTEMPTS = 3

# Заказ создаётся только этим инструментом (FR-9).
ORDER_TOOL = "create_service_order"

SYSTEM = (
    "Ты ассистент сервисной службы компании «ПолярТех». Отвечай на вопрос пользователя по-русски, коротко.\n"
    "Правила:\n"
    "- Факты о деталях, ошибках, регламентах и заменах бери только из инструментов. Не придумывай коды.\n"
    "- Остаток на складе и цену спрашивай у ERP через check_stock и get_price, не вычисляй сам.\n"
    "- Чтобы узнать номер детали по ошибке и модели, сначала вызови search_knowledge.\n"
    "- В search_knowledge и summarize_topic передавай вопрос пользователя целиком, сохраняя код ошибки,\n"
    "  модель, узел и деталь. Сокращай его только за счёт частей про склад, количество и цену.\n"
    "- Если пользователь просит оформить заказ и назвал количество, склад и модель, вызови\n"
    "  create_service_order. Нельзя писать, что заказ оформлен, без вызова этого инструмента.\n"
    "- Отказ search_knowledge не значит, что детали нет: если в facts есть номер детали вида P-…,\n"
    "  сразу вызови get_price и check_stock с этим номером.\n"
    "- Только если ни ответ, ни facts не дали данных, напиши, что в базе знаний не найдено.\n"
    "Не пересказывай пользователю служебные сообщения об ошибках инструментов."
)

# Подсказка, которую модель получает вместе с ошибкой вызова (FR-7).
_SEARCH_HINT = (
    "Номер детали должен быть в вопросе пользователя или в результатах предыдущих инструментов. "
    "Сначала вызови search_knowledge, чтобы узнать номер детали."
)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    sources: list[str] = []
    done: dict[str, dict] = {}
    known_text = question
    order_seen: dict | None = None
    calls = 0
    corrected = False

    while True:
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(reply.as_message())

        if not reply.tool_calls:
            answer = reply.content.strip()
            if _no_data(steps) and not answer.lower().startswith(REFUSAL.lower()):
                # ERP ответила «такой детали нет»: данных для ответа нет (FR-10).
                return AgentResult(
                    REFUSAL,
                    steps=steps,
                    sources=sources,
                    refused=True,
                    needs_confirmation=order_seen is not None,
                    pending_action=order_seen,
                )
            if _answer_is_grounded(answer, question, known_text, steps):
                return AgentResult(
                    answer,
                    steps=steps,
                    sources=sources,
                    refused=answer.lower().startswith("в базе знаний не найдено"),
                    needs_confirmation=order_seen is not None,
                    pending_action=order_seen,
                )
            # Расхождение с вопросом и результатами инструментов: одна просьба исправить (FR-10).
            if corrected:
                return AgentResult(
                    REFUSAL, steps=steps, sources=sources, refused=True,
                    needs_confirmation=order_seen is not None, pending_action=order_seen,
                )
            corrected = True
            messages.append({"role": "user", "content": _correction_message(question, known_text)})
            continue

        if calls >= max_steps:
            # Лимит шагов исчерпан: возвращаем то, что успели узнать (FR-7).
            return AgentResult(
                _interrupted(steps),
                steps=steps,
                sources=sources,
                refused=False,
                needs_confirmation=order_seen is not None,
                pending_action=order_seen,
            )

        for call in reply.tool_calls:
            calls += 1
            step, detail, payload = _run_call(
                call, tools, question, known_text, done, confirm, llm
            )
            steps.append(step)
            if step.result is not None:
                known_text = f"{known_text}\n{json.dumps(step.result, ensure_ascii=False)}"
                for source in _sources_of(step.result):
                    if source not in sources:
                        sources.append(source)
            if step.tool == ORDER_TOOL and step.error is None and step.result is not None:
                order_seen = None
            if payload is not None:
                order_seen = payload
            messages.append(tool_message(call, detail))
            if order_seen is not None:
                # Заказ без подтверждения: цикл останавливается, решение за пользователем (FR-9).
                return AgentResult(
                    _pending_answer(order_seen),
                    steps=steps,
                    sources=sources,
                    needs_confirmation=True,
                    pending_action=order_seen,
                )


def _run_call(
    call,
    tools: dict[str, Callable[..., dict]],
    question: str,
    known_text: str,
    done: dict[str, dict],
    confirm: bool,
    llm: LLMClient,
) -> tuple[ToolStep, dict, dict | None]:
    """Проверяет и выполняет один вызов. Возвращает шаг, ответ модели и отложенный заказ."""
    name, args = call.name, dict(call.arguments)

    if "_raw" in args:
        error = "аргументы вызова не разобрались как JSON"
        return ToolStep(name, args, error=error), {"error": error}, None
    if name not in tools or name not in {schema["function"]["name"] for schema in TOOL_SCHEMAS}:
        error = f"неизвестный инструмент «{name}»"
        return ToolStep(name, args, error=error), {"error": error}, None

    key = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
    if key in done:
        # Повторный вызов с теми же аргументами получает прежний результат (FR-7).
        previous = done[key]
        return ToolStep(name, args, result=previous), previous, None

    part_id = args.get("part_id")
    if part_id is not None and not _part_is_known(str(part_id), question, known_text):
        error = f"номер детали {part_id} нигде не назван. {_SEARCH_HINT}"
        return ToolStep(name, args, error=error), {"error": error}, None

    if name == ORDER_TOOL and not _qty_is_named(args.get("qty"), question):
        error = "количество в заказе должно быть названо пользователем. Спроси, сколько штук заказать."
        return ToolStep(name, args, error=error), {"error": error}, None

    if name == ORDER_TOOL and not confirm:
        action = {"tool": ORDER_TOOL, "args": args}
        detail = {"status": "ждёт подтверждения", "action": action}
        # Заказ не создан: шаг помечается ошибкой, иначе проверка оценки сочтёт его выполненным.
        return ToolStep(name, args, result=detail, error="заказ ждёт подтверждения пользователя"), detail, action

    try:
        result = _with_retries(tools[name], args)
    except ErpUnavailable as exc:
        error = f"данные ERP получить не удалось: {exc}"
        return ToolStep(name, args, error=error), {"error": error}, None
    except TypeError as exc:
        error = f"неверные аргументы вызова: {exc}"
        return ToolStep(name, args, error=error), {"error": error}, None

    done[key] = result
    step = ToolStep(name, args, result=result, error=result.get("error"))
    return step, result, None


def _with_retries(tool: Callable[..., dict], args: dict) -> dict:
    """Три попытки при `ErpUnavailable`: первая и ещё две (FR-8)."""
    last: Exception | None = None
    for _ in range(ERP_ATTEMPTS):
        try:
            return tool(**args)
        except ErpUnavailable as exc:
            last = exc
    raise ErpUnavailable(str(last))


def _part_is_known(part_id: str, question: str, known_text: str) -> bool:
    """Номер детали встречается в вопросе или в результатах прошлых инструментов (FR-7)."""
    code = normalize_code(part_id)
    if code is None:
        return False
    return code in {normalize_code(raw) for raw in find_codes(question) + find_codes(known_text)}


def _qty_is_named(qty, question: str) -> bool:
    """Количество заказа названо пользователем, а не выдумано моделью (FR-7)."""
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        return False
    numbers = {int(raw) for raw in re.findall(r"\d+", question)}
    return qty in numbers


def _sources_of(result: dict | None) -> list[str]:
    """Документы-источники из результата инструмента."""
    if not isinstance(result, dict):
        return []
    return [source for source in result.get("sources", []) if isinstance(source, str)]


def _no_data(steps: list[ToolStep]) -> bool:
    """ERP ответила, что такой детали нет, и другого ответа ERP не было: данных нет (FR-10)."""
    erp_steps = [step for step in steps if step.tool in {"check_stock", "get_price", "create_service_order"}]
    if not erp_steps or any(step.error is None for step in erp_steps):
        return False
    return any(step.result is not None and step.result.get("reason") == NOT_FOUND for step in erp_steps)


def _answer_is_grounded(answer: str, question: str, known_text: str, steps: list[ToolStep]) -> bool:
    """Каждый код, цена и остаток ответа есть в данных, а не выдуман пересказом (FR-10).

    Номера деталей берутся из вопроса и из всех результатов инструментов: факты графа
    и ответ ERP равноценны. А вот цена и остаток — это только данные ERP: поиск по графу
    цен не знает, и его текст с числом «1500 руб» доверять нельзя.
    """
    allowed_codes = set(find_codes(question)) | set(find_codes(known_text))
    if any(code not in allowed_codes for code in find_codes(answer)):
        return False
    tool_numbers = set(re.findall(r"\d+", question)) | set(re.findall(r"\d+", _erp_text(steps)))
    for raw in re.findall(r"(?:\d{1,3}(?:[  ]\d{3})+|\d+)\s*(?:руб|₽|шт)", answer):
        if re.sub(r"\D", "", raw) not in tool_numbers:
            return False
    return True


def _erp_text(steps: list[ToolStep]) -> str:
    """Результаты обращений в ERP: единственный источник цен и остатков для ответа."""
    results = [
        step.result
        for step in steps
        if step.tool in {"check_stock", "get_price", "create_service_order"} and step.result is not None
    ]
    return json.dumps(results, ensure_ascii=False)


def _correction_message(question: str, known_text: str) -> str:
    """Просьба исправить ответ: в нём есть то, чего нет в вопросе и результатах инструментов."""
    return (
        "В твоём ответе есть код, цена или остаток, которых нет ни в вопросе, ни в результатах "
        f"инструментов. Вопрос: «{question}». Используй только эти данные:\n{known_text}\n"
        "Напиши ответ заново, коротко."
    )


def _interrupted(steps: list[ToolStep]) -> str:
    """Ответ при исчерпанном лимите шагов: что успели узнать и что работа прервана."""
    facts = [step.result for step in steps if isinstance(step.result, dict)]
    answers = [item["answer"] for item in facts if isinstance(item.get("answer"), str) and item["answer"].strip()]
    tail = " Работа прервана: исчерпан лимит шагов."
    if answers:
        return f"{answers[-1]}{tail}"
    return f"Не удалось собрать ответ.{tail}"


def _pending_answer(action: dict) -> str:
    """Ответ при остановке на подтверждении заказа."""
    args = action.get("args", {})
    return (
        f"Готов оформить заказ: {args.get('qty')} шт. детали {args.get('part_id')} "
        f"на склад {args.get('warehouse')} для модели {args.get('model')}. Подтвердите заказ."
    )
