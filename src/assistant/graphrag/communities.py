"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

# Фиксированное зерно: разбиение не должно меняться от запуска к запуску.
SEED = 0

# Длина сводки в словах, заданная требованием FR-3.
MAX_WORDS = 150

# Сколько слов оставляем модели: остальное место в сводке занимают цифры, посчитанные кодом.
PROSE_WORDS = 100

# Сколько самых тяжёлых связей показывать модели: этого хватает, чтобы описать сообщество.
TOP_RELATIONS = 25


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []
    undirected = nx.Graph()
    undirected.add_nodes_from(graph.nodes)
    for source, target, attrs in graph.edges(data=True):
        weight = attrs.get("weight", 1)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
        else:
            undirected.add_edge(source, target, weight=weight)
    groups = nx.algorithms.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes[0])
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    Ключевые цифры дописываются к сводке дословно: модель склонна их пересказывать с потерями,
    а на них держатся ответы режима global.
    """
    facts = _facts(community, graph)
    messages = [
        {
            "role": "system",
            "content": (
                "Ты пишешь короткую сводку сообщества графа знаний сервисной службы «ПолярТех».\n"
                f"Ответ — до {PROSE_WORDS} слов, простым текстом, без списков и без markdown.\n"
                "Опирайся только на подсчёты и связи, которые даны ниже. Числа не придумывай:\n"
                "если числа нет в данных, не называй его.\n"
            ),
        },
        {"role": "user", "content": facts},
    ]
    reply = llm.chat(messages, max_tokens=512)
    keys = _key_figures(community, graph)
    # Цифры идут первыми: если сводка не помещается в 150 слов, обрезается пересказ модели,
    # а не посчитанные кодом числа, на которых держатся ответы режима global.
    room = max(MAX_WORDS - len(keys.split()), 0)
    prose = _shorten(reply.content, room)
    return _shorten(f"{keys}\n{prose}" if prose else keys, MAX_WORDS)


def _key_figures(community: Community, graph: nx.DiGraph) -> str:
    """Короткие цифры сообщества, посчитанные кодом: их нельзя терять при пересказе."""
    members = _scope(community, graph)
    parts = []
    lines = _line_groups(members, graph)
    if lines:
        parts.append(lines[0])
    for line in _failure_totals(_scoped_edges(members, graph), graph):
        parts.append(line)
    for line in _removed_parts(members, graph):
        parts.append(line)
    return " ".join(parts) if parts else "сводных чисел нет."


def _scope(community: Community, graph: nx.DiGraph) -> list[str]:
    """Узлы сообщества плюс его линейка: модели той же линейки и их детали.

    Сообщество графа не совпадает с линейкой оборудования: одна модель или деталь
    регулярно попадает в соседнее сообщество. Для обзорного вопроса по линейке нужен
    весь её состав, иначе модель молчит об отказах и заменах.
    """
    members = {node for node in community.nodes if node in graph}
    lines = {
        target
        for node in members
        if graph.nodes[node].get("type") == "Model"
        for _, target, attrs in graph.out_edges(node, data=True)
        if attrs.get("type") == "BELONGS_TO"
    }
    for node in graph.nodes:
        if graph.nodes[node].get("type") != "Model":
            continue
        if any(
            attrs.get("type") == "BELONGS_TO" and target in lines
            for _, target, attrs in graph.out_edges(node, data=True)
        ):
            members.add(node)
            members.update(
                target
                for _, target, attrs in graph.out_edges(node, data=True)
                if attrs.get("type") == "HAS_PART"
            )
    return sorted(members)


def _scoped_edges(members: list[str], graph: nx.DiGraph) -> list[tuple[str, str, str, int]]:
    """Связи, которые начинаются в узлах сообщества, даже если второй конец в другом сообществе.

    Отказы и замены почти всегда выходят за пределы сообщества: узел отказа и деталь-замена
    живут в соседних сообществах. Считать только внутренние связи нельзя — тогда сводка
    линейки молчит об отказах, и обзорный вопрос остаётся без ответа.
    """
    return [
        (source, attrs.get("type", ""), target, attrs.get("weight", 1))
        for source, target, attrs in graph.edges(data=True)
        if source in members
    ]


def _facts(community: Community, graph: nx.DiGraph) -> str:
    """Готовые подсчёты для модели: состав по типам, самые тяжёлые связи, итоги."""
    members = _scope(community, graph)
    inside = set(members)
    edges = [
        (source, kind, target, attrs.get("weight", 1))
        for source, target, attrs in graph.edges(data=True)
        if source in inside and target in inside
        for kind in (attrs.get("type", ""),)
    ]
    by_type = Counter(graph.nodes[node].get("type", "?") for node in members)
    by_relation = Counter(kind for _, kind, _, _ in edges)

    lines = [f"Сообщество {community.community_id}: узлов {len(members)}, связей {len(edges)}."]
    lines.append("Состав по типам: " + ", ".join(f"{kind} — {count}" for kind, count in sorted(by_type.items())) + ".")
    if by_relation:
        lines.append(
            "Связи по типам: "
            + ", ".join(f"{kind} — {count}" for kind, count in sorted(by_relation.items()))
            + "."
        )
    lines.extend(_line_groups(members, graph))
    lines.extend(_failure_totals(edges, graph))
    lines.extend(_removed_parts(members, graph))
    lines.append("Самые тяжёлые связи (вес — число документов, где связь встречается):")
    for source, kind, target, weight in sorted(edges, key=lambda item: (-item[3], item[0], item[2]))[:TOP_RELATIONS]:
        lines.append(f"- {source} -{kind}-> {target}, вес {weight}")
    lines.append(f"Всего وزن связей: {sum(weight for _, _, _, weight in edges)}.")
    return "\n".join(lines)


def _line_groups(members: list[str], graph: nx.DiGraph) -> list[str]:
    """К какой линейке относится каждая модель сообщества: без этого обзорный вопрос не отличить."""
    groups: dict[str, list[str]] = {}
    for node in members:
        if graph.nodes[node].get("type") != "Model":
            continue
        for _, target, attrs in graph.out_edges(node, data=True):
            if attrs.get("type") == "BELONGS_TO":
                groups.setdefault(graph.nodes[target].get("name", target), []).append(graph.nodes[node]["name"])
    if not groups:
        return []
    return [
        "Модели по линейкам: "
        + "; ".join(f"«{line}» — {', '.join(sorted(models))}" for line, models in sorted(groups.items()))
        + "."
    ]


def _failure_totals(edges: list[tuple[str, str, str, int]], graph: nx.DiGraph) -> list[str]:
    """Суммарный вес отказов по узлам и по моделям: эти числа отвечают на вопрос «чаще всего»."""
    by_component: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    models_of: dict[str, set[str]] = {}
    for source, kind, target, weight in edges:
        if kind != "FAILED":
            continue
        component = graph.nodes[target]["name"]
        by_component[component] += weight
        by_model[graph.nodes[source]["name"]] += weight
        models_of.setdefault(component, set()).add(graph.nodes[source]["name"])
    lines = []
    if by_component:
        ranked = ", ".join(
            f"{name} — {weight} (модели: {', '.join(sorted(models_of[name]))})"
            for name, weight in by_component.most_common()
        )
        lines.append(f"Суммарный вес отказов по узлам: {ranked}.")
    if by_model:
        ranked = ", ".join(f"{name} — {weight}" for name, weight in by_model.most_common())
        lines.append(f"Суммарный вес отказов по моделям: {ranked}.")
    return lines


def _removed_parts(members: list[str], graph: nx.DiGraph) -> list[str]:
    """Детали сообщества, снятые с производства, и деталь-замена."""
    removed = []
    for node in sorted(members):
        if graph.nodes[node].get("type") != "Part":
            continue
        replacements = sorted(
            graph.nodes[target]["name"]
            for target in graph.successors(node)
            if graph.edges[node, target].get("type") == "REPLACED_BY"
        )
        if replacements:
            removed.append(f"{graph.nodes[node]['name']} (заменена на {', '.join(replacements)})")
    return ["Снятые с производства детали: " + "; ".join(removed) + "."] if removed else []


def _shorten(text: str, limit: int = MAX_WORDS) -> str:
    """Обрезает текст до лимита слов: требование FR-3 ограничивает сводку 150 словами."""
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit])
