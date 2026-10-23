"""Small deterministic tools; no eval, shell, or arbitrary code execution."""
import ast
from datetime import datetime, timedelta, timezone
import math
import operator

from bot.brain.discord_tools import tool

UTILITY_TOOL_DEFINITIONS = [
    tool("calculate", "Calcula uma expressão aritmética com +, -, *, /, %, ** e parênteses. Use para cálculos em vez de adivinhar.",
         {"expression": {"type": "string", "description": "Expressão numérica, por exemplo (120 * 0.15) + 5."}}, ["expression"]),
    tool("convert_units", "Converte unidades de comprimento, massa, velocidade e temperatura. Não converte moedas.",
         {"value": {"type": "number"}, "from_unit": {"type": "string", "description": "m, km, cm, mi, ft, in; kg, g, lb, oz; m/s, km/h, mph; C, F, K."}, "to_unit": {"type": "string"}},
         ["value", "from_unit", "to_unit"]),
    tool("get_current_time", "Consulta data e hora atuais. O padrão é UTC-3 (São Paulo). Aceita deslocamento UTC em minutos; não resolve regras de horário de verão de uma cidade.",
         {"utc_offset_minutes": {"type": "integer", "minimum": -720, "maximum": 840, "description": "-180 para UTC-3, 0 para UTC."}}),
]


def calculate(expression):
    if len(expression) > 200:
        raise ValueError("Expressão longa demais.")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expressão complexa demais.")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow}
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10:
                raise ValueError("Expoente excede o limite de 10.")
            value = ops[type(node.op)](left, right)
        else:
            raise ValueError("Use apenas números e operadores aritméticos.")
        if not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e15:
            raise ValueError("Resultado fora do limite numérico.")
        return value
    return visit(tree.body)


def convert_units(value, from_unit, to_unit):
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError("Valor inválido.")
    source, target = from_unit.strip().lower(), to_unit.strip().lower()
    groups = [
        {"m": 1, "km": 1000, "cm": 0.01, "mi": 1609.344, "ft": 0.3048, "in": 0.0254},
        {"kg": 1, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
        {"m/s": 1, "km/h": 1/3.6, "mph": 0.44704},
    ]
    temperatures = {"c", "f", "k"}
    if source in temperatures and target in temperatures:
        celsius = (value - 32) * 5/9 if source == "f" else value - 273.15 if source == "k" else value
        return celsius * 9/5 + 32 if target == "f" else celsius + 273.15 if target == "k" else celsius
    for group in groups:
        if source in group and target in group:
            return value * group[source] / group[target]
    raise ValueError("Unidades desconhecidas ou incompatíveis.")


def execute_utility_tool(name, args):
    if name == "calculate":
        return {"expression": args["expression"], "result": calculate(args["expression"])}
    if name == "convert_units":
        return {**args, "result": convert_units(args["value"], args["from_unit"], args["to_unit"])}
    if name == "get_current_time":
        offset = args.get("utc_offset_minutes", -180)
        if not -720 <= offset <= 840:
            raise ValueError("Deslocamento UTC fora do intervalo permitido.")
        return {"time": datetime.now(timezone(timedelta(minutes=offset))).isoformat(), "utc_offset_minutes": offset}
    raise ValueError("Ferramenta utilitária desconhecida.")
