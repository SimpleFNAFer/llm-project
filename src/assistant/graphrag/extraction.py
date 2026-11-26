"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from typing import Any, Literal

from pydantic import ValidationError, create_model

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, CODE_TYPES
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import ENTITY_TYPES, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Chunk, ExtractionResult, Entity, Relation


def _annotation_from_schema(schema: dict, name: str) -> Any:
    """Преобразует типы, используемые в схеме извлечения, в аннотации Pydantic."""
    match schema["type"]:
        case "string":
            return Literal[*schema["enum"]] if "enum" in schema else str
        case "array":
            return list[_annotation_from_schema(schema["items"], f"{name}Item")]
        case "object":
            required = set(schema.get("required", ()))
            fields = {
                field: (_annotation_from_schema(spec, f"{name}_{field}"), ... if field in required else None)
                for field, spec in schema["properties"].items()
            }
            return create_model(name, **fields)
        case other:
            raise ValueError(f"Неподдерживаемый тип в схеме извлечения: {other}")


ExtractionResponse = _annotation_from_schema(EXTRACTION_SCHEMA, "ExtractionResponse")


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: проверки по тексту чанка и словарю

    prompt = f"""
    Найди сущности и связи чанка строго по схеме графа. Не додумывай.
    
    Список сущностей для извлечения: {ENTITY_TYPES}. Извлекай только их.
    Описания связей для извлечения: {RELATION_SIGNATURES}. Связи имеют направления. Определяй только эти типы связей. Примеры связей:
    - E-17 INDICATES пусковое реле;
    - AR-210 HAS_PART P-1042.

    Текст чанка:

    {chunk.text}
    """

    retry_instruction = "\n\nПредыдущий ответ не прошёл проверку. Верни только JSON по заданной схеме."
    for instruction in ("", retry_instruction):
        messages = [{"role": "user", "content": prompt + instruction}]
        resp = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
        print(resp.content)
        try:
            _validated_response = ExtractionResponse.model_validate_json(resp.content)
        except ValidationError as e:
            print(e)
            continue
        break
    else:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)

    codes = find_codes(chunk.text)

    result = ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        dropped=0,
        entities=[],
        relations=[],
    )

    for entity in _validated_response.entities:
        can_name = canonical_name(
            entity_type=entity.type, 
            raw=entity.name, 
            vocabulary=vocabulary(),
        )
        if can_name is None or entity.type in CODE_TYPES and can_name not in codes:
            result.dropped += 1
        else:
            result.entities.append(Entity(type=entity.type, name=can_name))
        
    for rel in _validated_response.relations:
        subj_name = canonical_name(
            entity_type=rel.subject_type, 
            raw=rel.subject, 
            vocabulary=vocabulary(),
        )
        subj_end = False
        if subj_name is not None:
            if rel.subject_type not in CODE_TYPES or subj_name in codes:
                subj_end = True
        if subj_end and not any(e.type == rel.subject_type and e.name == subj_name for e in result.entities):
            result.entities.append(Entity(type=rel.subject_type, name=subj_name))
        
        obj_name = canonical_name(
            entity_type=rel.object_type, 
            raw=rel.object, 
            vocabulary=vocabulary(),
        )
        obj_end = False
        if obj_name is not None:
            if rel.object_type not in CODE_TYPES or obj_name in codes:
                obj_end = True
        if obj_end and not any(e.type == rel.object_type and e.name == obj_name for e in result.entities):
            result.entities.append(Entity(type=rel.object_type, name=obj_name))
        
        if not subj_end or not obj_end:
            result.dropped += 1
        else:
            result.relations.append(Relation(
                Entity(type=rel.subject_type, name=subj_name),
                rel.type, 
                Entity(type=rel.object_type, name=obj_name)),
            )
        
    return result

