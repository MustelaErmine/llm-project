"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph()
    node_sources: dict[str, set[str]] = {}
    edge_sources: dict[tuple[str, str, str], set[str]] = {}

    for result in results:
        for entity in result.entities:
            node_sources.setdefault(entity.node_id, set()).add(result.chunk_id)
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            key = (relation.subject.node_id, relation.type, relation.object.node_id)
            edge_sources.setdefault(key, set()).add(result.chunk_id)
            # Концы связи — тоже узлы графа, даже если извлечение не перечислило их отдельно.
            for end in (relation.subject, relation.object):
                node_sources.setdefault(end.node_id, set()).add(result.chunk_id)
                if end.node_id not in graph:
                    graph.add_node(end.node_id, type=end.type, name=end.name, sources=[])

    conflicts = _drop_conflicting_replacements(edge_sources)
    for node_id, sources in node_sources.items():
        graph.nodes[node_id]["sources"] = sorted(sources)
    for (subject_id, kind, object_id), sources in edge_sources.items():
        graph.add_edge(subject_id, object_id, type=kind, weight=len(sources), sources=sorted(sources))

    graph.graph["conflicts"] = conflicts
    return graph


def _drop_conflicting_replacements(edge_sources: dict[tuple[str, str, str], set[str]]) -> int:
    """Разбирает встречные `REPLACED_BY` между двумя деталями и возвращает число конфликтов.

    Остаётся замена с большим весом, при равных весах удаляются обе: граф не должен утверждать
    одновременно, что старая деталь заменена новой, а новая — старой.
    """
    conflicts = 0
    pairs = set()
    for subject_id, kind, object_id in edge_sources:
        if kind == "REPLACED_BY" and subject_id != object_id:
            pairs.add(frozenset((subject_id, object_id)))
    for pair in pairs:
        first, second = sorted(pair)
        forward = (first, "REPLACED_BY", second)
        backward = (second, "REPLACED_BY", first)
        if forward not in edge_sources or backward not in edge_sources:
            continue
        conflicts += 1
        weight_forward = len(edge_sources[forward])
        weight_backward = len(edge_sources[backward])
        if weight_forward > weight_backward:
            del edge_sources[backward]
        elif weight_backward > weight_forward:
            del edge_sources[forward]
        else:
            del edge_sources[forward]
            del edge_sources[backward]
    return conflicts
