"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

from assistant.infra.grounding import normalize_code, code_type

from assistant.graphrag.search import local_search, global_search

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Выполняет local_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    }
                },
                "required": ["question"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Выполняет global_search",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    }
                },
                "required": ["question"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Проверяет наличие детали на складе и признак discontinued",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string"
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
                    }
                },
                "required": ["part_id", "warehouse"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Получает цену детали в рублях и признак discontinued",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string"
                    }
                },
                "required": ["part_id"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создаёт заказ и возвращает его номер",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string"
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
                    },
                    "qty": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "model": {
                        "type": "string"
                    }
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            }
        }
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP

    def search_knowledge(question: str) -> dict:
        result = local_search(question=question, kb=kb, llm=llm)
        return {
            "answer": result.text,
            "facts": result.facts,
            "sources": result.sources,
            "refused": result.refused,
        }
    
    def summarize_topic(question: str) -> dict:
        result = global_search(question=question, kb=kb, llm=llm)
        return {
            "answer": result.text,
            "sources": result.sources,
            "refused": result.refused,
        }
    
    def check_stock(part_id: str, warehouse: str):
        if warehouse not in erp.warehouses():
            return {"error": f"Указанный склад ({warehouse}) не существует или недоступен"}
        if normalize_code(part_id) is None or code_type(part_id) != "Part":
            return {"error": f"Некорректный код детали ({part_id})"}
        try:
            result = erp.get_stock(part_id=part_id, warehouse=warehouse)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP ответил отказом: {e}"}
        return result
    
    def get_price(part_id: str):
        if normalize_code(part_id) is None or code_type(part_id) != "Part":
            return {"error": f"Некорректный код детали ({part_id})"}
        try:
            result = erp.get_price(part_id=part_id)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP ответил отказом: {e}"}
        return result

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str):
        if normalize_code(model) is None or code_type(model) != "Model":
            return {"error": f"Некорректный код модели ({model})"}
        if qty <= 0:
            return {"error": f"Нулевое или отрицательное количество деталей (qty={qty})"}
        if warehouse not in erp.warehouses():
            return {"error": f"Указанный склад ({warehouse}) не существует или недоступен"}
        if normalize_code(part_id) is None or code_type(part_id) != "Part":
            return {"error": f"Некорректный код детали ({part_id})"}
        try:
            result = erp.create_order(part_id=part_id, warehouse=warehouse, qty=qty, model=model)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP ответил отказом: {e}"}
        return result

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
