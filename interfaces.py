# interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable, TypeVar, Type, Tuple

AgentId = TypeVar('AgentId', str, int)

class Policy:
    """
    Представляет политику, применяемую правительством.
    Значение политики может вычисляться динамически на основе безопасного
    Python-выражения, сгенерированного агентом.

    Атрибуты:
        id (str): Уникальный идентификатор политики.
        description (str): Текстовое описание сути политики.
        policy_type (str): Категория политики (например, 'set_tax_rate').
                           Используется моделью для определения способа применения.
        value_expression (str): Строка, содержащая Python-выражение для
                                вычисления значения политики.
        target_selector (str): Правило, определяющее, к каким агентам
                               (или части экономики) применяется политика.
                               (Например, 'default'). Пока не используется.
        _compiled_safe_code (Optional[Callable]): Скомпилированный байт-код
                                                  выражения после валидации.
                                                  Не предназначен для прямого
                                                  использования и сериализации.
    """
    def __init__(self,
                 id: str,
                 description: str,
                 policy_type: str,
                 value_expression: str,
                 target_selector: str = "default",
                 _compiled_safe_code: Optional[Callable] = None):
        self.id = id
        self.description = description
        self.policy_type = policy_type
        self.value_expression = value_expression
        self.target_selector = target_selector
        self._compiled_safe_code = _compiled_safe_code

    def __repr__(self):
        # ! Не включаем _compiled_safe_code для краткости
        return (f"Policy(id='{self.id}', type='{self.policy_type}', "
                f"target='{self.target_selector}', expr='{self.value_expression}')")

    def to_dict(self) -> Dict[str, Any]:
        """Возвращает словарь, представляющий политику (без скомпилированного кода)."""
        # Используется для безопасной сериализации
        return {
            "id": self.id,
            "description": self.description,
            "policy_type": self.policy_type,
            "value_expression": self.value_expression,
            "target_selector": self.target_selector
            # ! _compiled_safe_code не включается
        }


class PolicyDescriptor:
    """
    Описывает один тип политики, который может быть применен к модели.
    Предоставляет метаинформацию для агента-правительства и механизмов валидации.

    Атрибуты:
        policy_type_id (str): Уникальный идентификатор типа политики.
        description (str): Человекочитаемое описание политики.
        target_variable_name (Optional[str]): Имя переменной состояния модели,
                                              на которую влияет политика.
        value_type (Type): Ожидаемый тип данных значения выражения.
        value_range (Optional[Tuple[Any, Any]]): Допустимый диапазон значений.
        available_context_vars (List[str]): Список имен переменных из контекста,
                                            которые можно использовать в выражении.
        constraints (Dict[str, Any]): Словарь для дополнительных правил. Примеры:
                                     {'min_change_interval': 5, # Минимальный интервал изменения
                                      'change_cost': 10.0,      # Стоимость изменения
                                      'requires_approval': True} # Требует доп. утверждения
    """
    def __init__(self,
                 policy_type_id: str,
                 description: str,
                 value_type: Type,
                 available_context_vars: List[str],
                 target_variable_name: Optional[str] = None,
                 value_range: Optional[Tuple[Any, Any]] = None,
                 constraints: Optional[Dict[str, Any]] = None):
        self.policy_type_id = policy_type_id
        self.description = description
        self.target_variable_name = target_variable_name
        self.value_type = value_type
        self.value_range = value_range
        self.available_context_vars = available_context_vars
        self.constraints = constraints if constraints is not None else {}

    def __repr__(self):
        return (f"PolicyDescriptor(id='{self.policy_type_id}', "
                f"desc='{self.description}', "
                f"type={self.value_type.__name__}, "
                f"range={self.value_range}, "
                f"constraints={self.constraints})")


class BaseEconomicSystem(ABC):
    """
    Абстрактный базовый класс для моделируемой экономической системы.
    Определяет основной интерфейс взаимодействия.
    """

    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """Инициализация модели с заданными параметрами."""
        self.current_step: int = 0               # Текущий шаг симуляции
        self.state: Dict[str, Any] = {}          # Полное состояние модели
        self.active_policies: List[Policy] = []  # Действующие политики
        self.history: List[Dict[str, Any]] = []  # История состояний

    @abstractmethod
    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        """
        Возвращает список дескрипторов для ВСЕХ типов политик,
        которые эта экономическая модель поддерживает.
        """
        pass

    @abstractmethod
    def step(self) -> None:
        """
        Выполняет один шаг симуляции экономической модели.
        Включает применение активных политик, обновление состояния агентов,
        расчет метрик и обновление общего состояния `self.state`.
        """
        pass

    @abstractmethod
    def get_state_for_agent(self) -> Dict[str, Any]:
        """
        Возвращает часть состояния системы, видимую агенту-правительству.
        """
        pass

    @abstractmethod
    def apply_policy_change(self, policy_change: Optional[Policy]) -> None:
        """
        Применяет изменение в активных политиках модели.
        Обновляет список `self.active_policies` на основе предложения агента.
        """
        pass

    @abstractmethod
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Возвращает словарь с ключевыми агрегированными метриками
        текущего состояния экономики.
        """
        pass

    @abstractmethod
    def emulate_policy(self,
                       policy: Policy,
                       duration: int,
                       agents_subset: Optional[List[AgentId]] = None
                       ) -> Dict[str, Any]:
        """
        **Метод для будущей реализации.**
        Выполняет "пробный прогон" предложенной политики без изменения
        основного состояния модели для оценки ее эффектов.

        Args:
            policy: Политика для эмуляции.
            duration: Количество шагов для эмуляции.
            agents_subset: Опциональный список ID агентов для эмуляции.

        Returns:
            Словарь с результатами эмуляции.
        """
        # Этот метод должен быть реализован в конкретных моделях экономики.
        raise NotImplementedError("Метод emulate_policy не реализован.")


class BaseGovernmentAgent(ABC):
    """
    Абстрактный базовый класс для агента-правительства.
    Определяет интерфейс принятия решений о политике.
    """

    @abstractmethod
    def __init__(self, params: Dict[str, Any]):
        """Инициализация агента с заданными параметрами."""
        pass

    @abstractmethod
    def decide_policy(self,
                      current_state_for_agent: Dict[str, Any],
                      history: List[Dict[str, Any]],
                      economic_system: BaseEconomicSystem,
                      llm_extra_context: Optional[Dict[str, Any]] = None
                      ) -> Optional[Policy]:
        """
        Основной метод принятия решений агента.
        Анализирует состояние, историю и контекст для генерации/обновления политики.

        Args:
            current_state_for_agent: Состояние системы, видимое агенту.
            history: История состояний системы.
            economic_system: Экземпляр текущей экономической модели.
            llm_extra_context: Опциональный словарь с доп. информацией для LLM.

        Returns:
            Объект `Policy` (если предложено изменение) или `None`.
        """
        pass