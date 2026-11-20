"""Инструменты агента. Требование FR-6."""

import json
from collections.abc import Callable

from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Четыре склада ERP. Перечень нужен и модели в описании инструмента, и проверке аргументов.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

# Причина ошибки ERP машинным кодом: агент различает «данных нет» и «ERP отказала» (FR-8).
NOT_FOUND = "not_found"
REJECTED = "rejected"

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Найти факты в графе знаний: коды ошибок, детали, узлы, поставщики, регламенты, замены. "
                "Вызывай первым, чтобы узнать номер детали или регламент."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Вопрос по документам сервисной службы"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Обзорный ответ по сводкам сообществ графа: что чаще отказывает, какие детали сняты "
                "с производства в линейке. Для вопросов «чаще всего», «в целом по линейке»."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Обзорный вопрос по линейке оборудования"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Остаток детали на складе и признак «снята с производства».",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
                    "warehouse": {
                        "type": "string",
                        "description": "Склад",
                        "enum": WAREHOUSES,
                    },
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Цена детали в рублях и признак «снята с производства».",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
                },
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": (
                "Создать заказ на деталь. Вызывай только когда пользователь назвал количество "
                "и подтвердил заказ."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали, например P-1042B"},
                    "warehouse": {
                        "type": "string",
                        "description": "Склад",
                        "enum": WAREHOUSES,
                    },
                    "qty": {"type": "integer", "description": "Количество штук, больше нуля"},
                    "model": {"type": "string", "description": "Код модели оборудования, например AR-210"},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search_knowledge(*, question: str = "", **_) -> dict:
        from assistant.graphrag.search import local_search

        if not question.strip():
            return {"error": "не указан вопрос для поиска по базе знаний"}
        answer = local_search(question, kb, llm)
        return {
            "answer": answer.text,
            "facts": answer.facts,
            "sources": answer.sources,
            "refused": answer.refused,
        }

    def summarize_topic(*, question: str = "", **_) -> dict:
        from assistant.graphrag.search import global_search

        if not question.strip():
            return {"error": "не указан вопрос для обзора по базе знаний"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(*, part_id: str = "", warehouse: str = "", **_) -> dict:
        error = _warehouse_error(warehouse) or _part_error(part_id)
        if error:
            return {"error": error}
        part = _part_code(part_id)
        try:
            return erp.get_stock(part, warehouse)
        except ErpNotFound as exc:
            return {"error": f"в ERP нет детали {part}: {exc}", "reason": NOT_FOUND}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос остатка: {exc}", "reason": REJECTED}

    def get_price(*, part_id: str = "", **_) -> dict:
        error = _part_error(part_id)
        if error:
            return {"error": error}
        part = _part_code(part_id)
        try:
            return erp.get_price(part)
        except ErpNotFound as exc:
            return {"error": f"в ERP нет детали {part}: {exc}", "reason": NOT_FOUND}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос цены: {exc}", "reason": REJECTED}

    def create_service_order(
        *, part_id: str = "", warehouse: str = "", qty: int = 0, model: str = "", **_
    ) -> dict:
        error = _warehouse_error(warehouse) or _part_error(part_id) or _qty_error(qty)
        if error:
            return {"error": error}
        part = _part_code(part_id)
        try:
            return erp.create_order(part, warehouse, qty, model)
        except ErpNotFound as exc:
            return {"error": f"в ERP нет детали {part}: {exc}", "reason": NOT_FOUND}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила заказ: {exc}", "reason": REJECTED}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }


def as_json(result: dict) -> str:
    """Результат инструмента одной строкой для сообщения модели."""
    return json.dumps(result, ensure_ascii=False)


def _part_code(raw: str) -> str:
    """Канонический номер детали: «p 1042b» -> «P-1042B»."""
    return normalize_code(raw) or raw.strip().upper()


def _part_error(raw: str) -> str | None:
    """Номер детали должен быть кодом детали вида P-1042 или P-1042B."""
    code = normalize_code(raw)
    if code is None or code_type(code) != "Part":
        return f"номер детали должен быть вида P-1042 или P-1042B, получено «{raw}»"
    return None


def _warehouse_error(raw: str) -> str | None:
    """Склад должен быть из перечня ERP."""
    if raw not in WAREHOUSES:
        return f"неизвестный склад «{raw}», допустимы: {', '.join(WAREHOUSES)}"
    return None


def _qty_error(qty) -> str | None:
    """Количество должно быть целым числом больше нуля."""
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        return f"количество должно быть целым числом больше нуля, получено «{qty}»"
    return None
