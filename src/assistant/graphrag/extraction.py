"""Извлечение сущностей и связей из чанка. Требование FR-1."""

from typing import Any, Literal

from pydantic import ValidationError, create_model

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import ENTITY_TYPES, EXTRACTION_SCHEMA, RELATION_SIGNATURES, Chunk, ExtractionResult


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

    codes = find_codes(chunk.text)
    names = find_names(chunk.text, vocabulary())

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
        try:
            _validated_response = ExtractionResponse.model_validate_json(resp.content)
        except ValidationError:
            continue
        break
    else:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)

