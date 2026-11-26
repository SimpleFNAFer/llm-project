"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult, Relation


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """

    graph = nx.DiGraph()
    graph.graph["conflicts"] = 0


    def upsert_graph(chunk_id: str, relation: Relation):
        if not relation.matches_schema():
            return
        
        for node in (relation.subject, relation.object):
            graph.add_node(
                node.node_id, 
                type=node.type, 
                name=node.name,
            )
            sources = graph.nodes[node.node_id].setdefault("sources", [])
            if chunk_id not in sources:
                sources.append(chunk_id)

        graph.add_edge(
            relation.subject.node_id, 
            relation.object.node_id,
            type=relation.type
        )
        edge = graph.edges[relation.subject.node_id, relation.object.node_id]
        sources = edge.setdefault("sources", [])
        if chunk_id not in sources:
            sources.append(chunk_id)
        edge["weight"] = len(sources)


    for result in results:
        chunk_id = result.chunk_id
        for relation in result.relations:
            upsert_graph(chunk_id=chunk_id, relation=relation)

    for subj, obj, data in list(graph.edges(data=True)):
        if not graph.has_edge(subj, obj):
            continue

        if data["type"] != "REPLACED_BY":
            continue

        if graph.has_edge(obj, subj):
            graph.graph["conflicts"] += 1
            
            if graph.edges[subj, obj]["weight"] == graph.edges[obj, subj]["weight"]:
                graph.remove_edge(subj, obj)
                graph.remove_edge(obj, subj)
            elif graph.edges[subj, obj]["weight"] > graph.edges[obj, subj]["weight"]:
                graph.remove_edge(obj, subj)
            else:
                graph.remove_edge(subj, obj)

    return graph