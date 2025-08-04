# govsim/utils/visualization_utils.py
from pathlib import Path
from govsim.config import LOGS_DIR

def get_log_file_path(filename: str) -> Path:
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