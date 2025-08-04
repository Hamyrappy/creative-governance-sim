# policy_utils.py
import ast
import math         # Разрешаем безопасные функции из стандартного модуля math
import numpy as np  # Разрешаем избранные безопасные функции из numpy
import builtins     # Импортируем модуль builtins явно
from typing import Dict, Any, Set, Callable, Optional

# --- Конфигурация безопасности ---
# (Списки ALLOWED_NODE_TYPES, ALLOWED_BUILTINS_NAMES, ALLOWED_MATH_NAMES, ALLOWED_NP_NAMES остаются без изменений)
# Разрешенные типы узлов AST (основные операции, константы, переменные, вызовы)
ALLOWED_NODE_TYPES = {
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
    ast.Call, ast.Attribute, # <<<--- ДОБАВЛЯЕМ ast.Attribute в разрешенные типы узлов
    # Операторы:
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, # Арифметика
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, # Сравнение
    ast.And, ast.Or, # Логические
    ast.USub, ast.UAdd, ast.Not, # Унарные
}

# Разрешенные имена встроенных функций/констант (передаются через safe_context)
ALLOWED_BUILTINS_NAMES = {
    'min', 'max', 'abs', 'round', 'pow', 'len', # Базовые
    'True', 'False', 'None', # Константы (обрабатываются парсером, но оставляем для ясности)
}

# Разрешенные имена из безопасных модулей (добавляются в safe_context)
ALLOWED_MATH_NAMES = {name for name in dir(math) if not name.startswith('_')}
ALLOWED_NP_NAMES = {'clip', 'mean', 'std', 'sum', 'array', 'sqrt', 'exp', 'log', 'log10'} # Только безопасные функции numpy!


# --- Классы исключений ---
class PolicyValidationError(ValueError):
    """Исключение, возникающее при ошибке валидации AST выражения политики."""
    pass

# --- AST Валидатор ---
class SafeExpressionVisitor(ast.NodeVisitor):
    """
    Класс для обхода AST дерева выражения и проверки его безопасности.
    Проверяет каждый узел на соответствие белым спискам типов и имен.
    """
    def __init__(self, allowed_names: Set[str]):
        self.allowed_names = allowed_names
        # Добавим множества имен из модулей для проверки в visit_Attribute
        self.allowed_math_funcs = ALLOWED_MATH_NAMES
        self.allowed_np_funcs = ALLOWED_NP_NAMES
        super().__init__()

    def visit(self, node: ast.AST):
        """Переопределенный метод visit для проверки типа узла перед обходом."""
        node_type = type(node)
        # Разрешаем только узлы из белого списка
        if node_type not in ALLOWED_NODE_TYPES:
            # Формируем более информативное сообщение об ошибке
            detail = f"Тип узла '{node_type.__name__}' не разрешен в выражениях политик."
            try:
                # Попытка получить строку кода, вызвавшего ошибку (может не сработать для всех узлов)
                problem_code = ast.unparse(node)
                detail += f" Проблемный код: '{problem_code}'"
            except Exception:
                 pass # Не удалось получить код - не страшно
            raise PolicyValidationError(detail)
        super().visit(node) # Продолжаем обход для дочерних узлов

    def visit_Name(self, node: ast.Name):
        """Проверяет использование имен (переменных, функций)."""
        if isinstance(node.ctx, ast.Load) and node.id not in self.allowed_names:
             # Проверяем, не является ли это разрешенным модулем (math, np)
             # Эти модули сами по себе не вызываются, но нужны для доступа к атрибутам
            if node.id not in ['math', 'np']:
                 raise PolicyValidationError(
                    f"Использование недопустимого имени '{node.id}' в выражении.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """
        Проверяет доступ к атрибутам (например, math.sin, np.clip).
        Разрешает доступ ТОЛЬКО к разрешенным функциям из разрешенных модулей (math, np).
        """
        # Узел node.value должен быть именем (ast.Name)
        if not isinstance(node.value, ast.Name):
            raise PolicyValidationError(f"Доступ к атрибутам разрешен только для модулей 'math' и 'np'. Найдено: {type(node.value).__name__}")

        module_name = node.value.id
        attr_name = node.attr

        # Проверяем, разрешен ли модуль и атрибут в нем
        if module_name == 'math':
            if attr_name not in self.allowed_math_funcs:
                raise PolicyValidationError(f"Доступ к атрибуту 'math.{attr_name}' запрещен.")
            # Если разрешено, рекурсивно посещаем узел 'math' (для проверки, что 'math' разрешен)
            self.visit(node.value)
        elif module_name == 'np':
            if attr_name not in self.allowed_np_funcs:
                raise PolicyValidationError(f"Доступ к атрибуту 'np.{attr_name}' запрещен.")
            # Посещаем узел 'np'
            self.visit(node.value)
        else:
            # Доступ к атрибутам других объектов запрещен
            raise PolicyValidationError(f"Доступ к атрибутам модуля или объекта '{module_name}' запрещен.")

        # Не вызываем generic_visit для атрибута, т.к. мы его уже проверили


# --- Основные функции утилиты ---

def validate_and_compile_policy_expression(
    expression: str,
    context_variable_names: Set[str]
) -> Callable:
    """
    Валидирует строку с Python-выражением и компилирует ее в безопасный байт-код.
    (Описание процесса остается тем же)
    """
    if not isinstance(expression, str) or not expression.strip():
         raise PolicyValidationError("Выражение политики не может быть пустым.")

    try:
        tree = ast.parse(expression.strip(), mode='eval')
    except SyntaxError as e:
        raise PolicyValidationError(f"Синтаксическая ошибка в выражении политики: {e}")
    except Exception as e:
        raise PolicyValidationError(f"Ошибка парсинга выражения политики: {e}")

    # Собираем полный набор разрешенных имен для этого выражения
    # Включаем имена самих модулей 'math' и 'np', чтобы visit_Attribute мог их проверить
    allowed_names = (
        ALLOWED_BUILTINS_NAMES |
        ALLOWED_MATH_NAMES | # Имена функций из math
        ALLOWED_NP_NAMES |   # Имена функций из np
        context_variable_names |
        {'math', 'np'}       # <<<--- ДОБАВЛЯЕМ ИМЕНА МОДУЛЕЙ
    )

    validator = SafeExpressionVisitor(allowed_names)
    try:
        validator.visit(tree)
    except PolicyValidationError:
         raise
    except Exception as e:
         raise PolicyValidationError(f"Неожиданная ошибка при валидации AST: {e}")

    try:
        compiled_code = compile(tree, filename="<policy_expression>", mode="eval")
        return compiled_code
    except Exception as e:
        raise PolicyValidationError(f"Ошибка компиляции безопасного выражения: {e}")


def evaluate_safe_policy_code(
    compiled_code: Callable,
    simulation_context: Dict[str, Any]
) -> Any:
    """
    Безопасно выполняет скомпилированный байт-код политики.
    (Описание процесса остается тем же)
    """
    # 1. Формируем безопасный глобальный контекст для eval
    safe_globals = {}

    # Добавляем разрешенные *встроенные ФУНКЦИИ* (исключаем True/False/None)
    for name in ALLOWED_BUILTINS_NAMES:
        if name in ['True', 'False', 'None']: continue # Константы обрабатывает AST
        builtin_func = getattr(builtins, name, None)
        if callable(builtin_func): # Убедимся, что это функция
            safe_globals[name] = builtin_func
        else:
            print(f"Предупреждение: Ожидалась функция, но '{name}' не является вызываемым в builtins.")

    # Добавляем разрешенные имена из math (передаем сами функции)
    for name in ALLOWED_MATH_NAMES:
         if hasattr(math, name):
             safe_globals[name] = getattr(math, name)

    # Добавляем разрешенные имена из numpy (передаем сами функции)
    for name in ALLOWED_NP_NAMES:
         if hasattr(np, name):
             safe_globals[name] = getattr(np, name)

    # <<<--- ВАЖНО: Добавляем сами модули math и np в globals --->>>
    # Это необходимо, чтобы выражения вида 'math.sin()' или 'np.clip()' работали
    safe_globals['math'] = math
    safe_globals['np'] = np

    # Добавляем переменные из текущего контекста симуляции
    safe_globals.update(simulation_context)

    # 2. Выполняем байт-код с безопасным окружением
    try:
        # Используем созданный safe_globals как глобальное пространство имен.
        # Передаем ПУСТОЙ словарь в __builtins__ внутри globals, чтобы заблокировать опасные.
        result = eval(compiled_code, {'__builtins__': {}, **safe_globals}, {}) # locals пуст
        return result
    except Exception as e:
        print(f"Ошибка ВЫПОЛНЕНИЯ выражения политики: {e}")
        return None