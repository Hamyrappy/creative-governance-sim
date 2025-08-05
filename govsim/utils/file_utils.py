# file_utils.py

import numpy as np
import pathlib
from pathlib import Path
from govsim.config import LOGS_DIR

def default_serializer(o): # Используется при формировании логов, например в simulation.py
    if isinstance(o, (np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(o)
    elif isinstance(o, (np.float16, np.float32, np.float64)):
        return float(o)
    elif isinstance(o, (np.ndarray,)): # Конвертируем массивы в списки
        return o.tolist()
    elif isinstance(o, (np.bool_)):
        return bool(o)
    elif isinstance(o, (np.void)): # Обработка void (если встретится)
        return None
    elif isinstance(o, pathlib.Path):
        return str(o)
    # TODO Можно добавить обработку datetime и т.д.
    print(f"Предупреждение: Не удалось сериализовать объект типа {type(o)}. Заменен на None.")
    return None # Заменяем несериализуемое на None


def get_log_file_path(filename: str) -> Path: # Используется в скриптах визуализации и анализа логов
    """
    Возвращает полный путь к файлу в директории логов проекта.
    Автоматически добавляет .json к названиям файлов, если формат .json не указан
    """
    if filename[-5:] != '.json':
        filename = filename + '.json'

    file_path = LOGS_DIR / filename
    
    if not LOGS_DIR.is_dir():
        # Эта проверка на случай, если сама папка logs бb=ыла удалена или потеряна
        raise FileNotFoundError(f"Директория логов не найдена по пути: {LOGS_DIR}")
        
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл лога '{filename}' не найден в директории: {LOGS_DIR}")

    return file_path