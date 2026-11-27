"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    groups = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=42)
    return [Community(community_id=i, nodes=sorted(nodes)) for i, nodes in enumerate(groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    edges = []

    for i in range(len(community.nodes)-1):
        for j in range(i+1, len(community.nodes)):
            ni = community.nodes[i]
            nj = community.nodes[j]
            if graph.has_edge(ni, nj) and graph.edges[ni, nj] not in edges:
                edges.append(graph.edges[ni, nj])
            elif graph.has_edge(nj, ni) and graph.edges[nj, ni] not in edges:
                edges.append(graph.edges[nj, ni])
    
    prompt = f"""
        Сформулируй описание сообщества графа не длиннее 150 слов.
        В описании отрази, какие сущности входят, как связаны, что встречается чаще.

        Данные о сообществе:
        {community}

        Связи сущностей в сообществе:
        {edges}
    """
    messages = [{"role": "user", "content": prompt}]

    return llm.chat(messages).content[:150]


