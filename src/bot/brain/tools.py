import os
import asyncio
import re
import json
import logging
from bot.config import config
from bot.brain.memory import MemoryManager
from bot.brain.discord_tools import DISCORD_TOOL_DEFINITIONS, execute_discord_tool
from bot.brain.web_tools import web_search, browse_webpage, public_url
from bot.brain.utility_tools import UTILITY_TOOL_DEFINITIONS, execute_utility_tool

logger = logging.getLogger("VoiceBot.Tools")

memory_manager = MemoryManager()

# Tool definitions in OpenAI format
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Pesquisa informações em tempo real na internet pelo DuckDuckGo. Use sempre que precisar de notícias, clima, dados recentes ou respostas a fatos externos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "O termo de pesquisa em português ou inglês."
                    },
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Quantidade de fontes, padrão 3."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browse_webpage",
            "description": "Lê o conteúdo textual principal de uma página web a partir de uma URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "A URL completa da página a ser lida."
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Salva uma informação importante, preferência ou fato aprendido sobre um usuário ou sobre o grupo para consulta futura.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "O fato ou informação a ser gravada."
                    },
                    "category": {
                        "type": "string",
                        "description": "Categoria do fato (ex: preferencia, biografia, projeto, piada interna)."
                    }
                },
                "required": ["fact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Consulta a memória persistente para lembrar preferências, detalhes passados ou fatos de um usuário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Termo para buscar nas memórias salvas."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": "Cria um novo arquivo de habilidade/procedimento em formato Markdown na pasta de skills para que o bot aprenda a executar novas tarefas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Nome da skill em minúsculas com underscores (ex: tradutor_tecnico, contador_piadas)."
                    },
                    "instructions": {
                        "type": "string",
                        "description": "Instruções completas detalhando como a habilidade deve ser executada."
                    }
                },
                "required": ["skill_name", "instructions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_skills",
            "description": "Lista todas as habilidades aprendidas disponíveis no bot.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]
TOOL_DEFINITIONS.extend(DISCORD_TOOL_DEFINITIONS)
TOOL_DEFINITIONS.extend(UTILITY_TOOL_DEFINITIONS)

async def _execute_tool(name: str, args: dict, context: dict) -> str:
    """Executa a ferramenta solicitada e retorna o resultado em string."""
    logger.info(f"Executing tool '{name}' with args: {args}")
    try:
        if name.startswith("discord_"):
            return execute_discord_tool(name, args, context)
        elif name in ("calculate", "convert_units", "get_current_time"):
            return json.dumps(execute_utility_tool(name, args), ensure_ascii=False)
        elif name == "web_search":
            return json.dumps(await web_search(args["query"], args.get("max_results", 3)), ensure_ascii=False)

        elif name == "browse_webpage":
            url = str(public_url(args["url"]))
            cache = context.setdefault("_web_page_cache", {})
            if url in cache:
                return cache[url]
            context.setdefault("_web_attempted_urls", set()).add(str(public_url(url).with_query(None)))
            try:
                result = json.dumps(await browse_webpage(url), ensure_ascii=False)
            except ValueError as error:
                status = re.search(r"HTTP (\d{3})", str(error))
                if not status:
                    raise
                logger.warning("Page unavailable HTTP %s: %s; searching alternative sources", status[1], url)
                result = json.dumps({"url": url, "error": str(error), "http_status": int(status[1]),
                                     "untrusted_content": True}, ensure_ascii=False)
            cache[url] = result
            return result

        elif name == "save_memory":
            fact = args.get("fact", "")
            category = args.get("category", "general")
            user_id = context.get("user_id")
            username = context.get("username", "")
            res = memory_manager.save_fact(fact, category, user_id, username)
            return res.get("message", "Memória gravada.")

        elif name == "recall_memory":
            query = args.get("query", "")
            user_id = context.get("user_id")
            memories = memory_manager.recall_facts(query, user_id)
            if not memories:
                return "Nenhuma memória encontrada para esta busca."
            return "Memórias encontradas:\n- " + "\n- ".join(memories)

        elif name == "create_skill":
            skill_name = args.get("skill_name", "").strip().lower().replace(" ", "_")
            if not re.fullmatch(r"[a-z0-9_]{1,64}", skill_name):
                return "Invalid skill name; use letters, numbers and underscores."
            instructions = args.get("instructions", "")
            os.makedirs(config.skills_dir, exist_ok=True)
            skill_path = os.path.join(config.skills_dir, f"{skill_name}.md")
            with open(skill_path, "w", encoding="utf-8") as f:
                f.write(f"# Skill: {skill_name}\n\n{instructions}\n")
            logger.info(f"Created new skill file at {skill_path}")
            return f"Habilidade '{skill_name}' criada com sucesso!"

        elif name == "list_skills":
            if not os.path.exists(config.skills_dir):
                return "Nenhuma habilidade personalizada criada ainda."
            files = [f for f in os.listdir(config.skills_dir) if f.endswith(".md")]
            if not files:
                return "Nenhuma habilidade personalizada encontrada."
            return "Habilidades disponíveis: " + ", ".join([f.replace(".md", "") for f in files])

        else:
            return f"Ferramenta desconhecida: {name}"

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return f"Erro ao executar {name}: {str(e)}"


async def execute_tool(name: str, args: dict, context: dict) -> str:
    definition = next((d["function"] for d in TOOL_DEFINITIONS if d["function"]["name"] == name), None)
    if definition is None:
        return "Ferramenta desconhecida."
    if not isinstance(args, dict):
        return "Os argumentos da ferramenta devem ser um objeto JSON."
    parameters = definition["parameters"]
    properties = parameters["properties"]
    if any(key not in properties for key in args):
        return "A ferramenta recebeu argumentos desconhecidos."
    if any(key not in args for key in parameters.get("required", [])):
        return "Faltam argumentos obrigatórios da ferramenta."
    for key, value in args.items():
        schema = properties[key]
        if schema.get("type") == "string" and (not isinstance(value, str) or not value.strip() or len(value) > 8000):
            return f"Argumento inválido: {key}. Use texto não vazio com no máximo 8000 caracteres."
        if schema.get("type") == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"Argumento inválido: {key}. Use um número inteiro."
        if schema.get("type") == "number" and type(value) not in (int, float):
            return f"Argumento inválido: {key}. Use um número."
        if "enum" in schema and value not in schema["enum"]:
            return f"Valor não permitido para {key}."
    try:
        async with asyncio.timeout(15):
            result = await _execute_tool(name, args, context)
            if len(result) > 12000:
                return json.dumps({"truncated": True, "excerpt": result[:10000]}, ensure_ascii=False)
            return result
    except TimeoutError:
        return "A ferramenta excedeu o tempo limite. Não foi possível obter os dados agora."
