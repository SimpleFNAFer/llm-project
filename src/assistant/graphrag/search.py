"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

from itertools import combinations, pairwise

import networkx as nx
from pydantic import BaseModel, Field, ValidationError

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def _replacement_edges(graph: nx.DiGraph, parts: set[str]) -> set[tuple[str, str]]:
    edges = set()
    pending = list(parts)
    seen = set(parts)

    while pending:
        old = pending.pop()
        for new, attrs in graph[old].items():
            if attrs.get("type") != "REPLACED_BY":
                continue
            edges.add((old, new))
            if new not in seen:
                seen.add(new)
                pending.append(new)

    return edges


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
    anchors = []
    codes = find_codes(question)
    anchors.extend(f"{code_type(code)}:{code}" for code in codes)
    names = find_names(question, kb.vocabulary)
    anchors.extend(f"{kind}:{name}" for kind, name in names)

    if len(anchors) == 0:
        return Answer(..., refused=True)

    for anchor in anchors:
        if anchor not in kb.graph.nodes:
            return Answer(..., refused=True)

    facts = []
    graph = kb.graph
    walk = graph.to_undirected(as_view=True)

    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(walk, anchors[0], cutoff=2)
        edges = graph.subgraph(nearby).edges(data=True)
        facts.extend(edges)
    else:
        edge_ids = set()
        parts_on_paths = set()
        for start, end in combinations(anchors, 2):
            for path in nx.all_simple_paths(walk, start, end, cutoff=3):
                parts_on_paths.update(node for node in path if graph.nodes[node].get("type") == "Part")
                for u, v in pairwise(path):
                    if graph.has_edge(u, v):
                        edge_ids.add((u, v))
                    if graph.has_edge(v, u):
                        edge_ids.add((v, u))
        edge_ids.update(_replacement_edges(graph, parts_on_paths))
        facts.extend((u, v, graph.edges[u, v]) for u, v in edge_ids)
    
    if len(facts) == 0:
        return Answer(..., refused=True)

    facts_str = [f"{i+1}. {facts[i]}" for i in range(len(facts))]

    sources = sorted({
        chunk_id.split("#", 1)[0]
        for _, _, attrs in facts
        for chunk_id in attrs.get("sources", [])
    })
    facts_text = "\n".join(facts_str)[:8000]

    prompt = f"""
        Ответь на вопрос строго в соответствии с предоставленными фактами.
        
        Вопрос: {question}

        Факты по номерам:
        {facts_text}
    """
    messages = [{"role": "user", "content": prompt}]

    return Answer(text=llm.chat(messages).content, sources=sources, facts=facts_str)


bounds = MAP_SCHEMA["properties"]["score"]


class MapResponse(BaseModel):
    answer: str
    score: int = Field(ge=bounds["minimum"], le=bounds["maximum"])


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    facts: list[str] = []
    sources: set[str] = set()

    for community in kb.communities:
        prompt = f"""
            Ответь на вопрос строго по схеме, используя только переданные данные. Дай ответу оценку полезности.

            Схема:
            {MAP_SCHEMA}

            Вопрос: {question}

            Сводка сообщества:
            {community.summary}
        """
        messages = [{"role": "user", "content": prompt}]
        response = llm.chat(messages, json_schema=MAP_SCHEMA)
        try:
            partial = MapResponse.model_validate_json(response.content, strict=True)
        except ValidationError:
            continue
        if partial.score <= 0:
            continue

        facts.append(partial.answer)
        for _, _, attrs in kb.graph.subgraph(community.nodes).edges(data=True):
            sources.update(chunk_id.split("#", 1)[0] for chunk_id in attrs.get("sources", []))

    if not facts:
        return Answer(..., refused=True)

    partial_answers = "\n".join(facts)
    prompt = f"""
        Ответь на вопрос, объединив приведённые частичные ответы.

        Вопрос: {question}

        Частичные ответы:
        {partial_answers}
    """
    messages = [{"role": "user", "content": prompt}]
    res = llm.chat(messages).content
    return Answer(text=res, facts=facts, sources=sorted(sources))
