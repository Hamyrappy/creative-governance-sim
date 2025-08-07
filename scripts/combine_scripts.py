import os

# Внутренняя константа для указания целевой папки
TARGET_DIRECTORY = "govsim"

# Имя выходного файла
OUTPUT_FILENAME = "combined_code.txt"

def get_all_py_files(directory):
    """
    Рекурсивно находит все .py файлы в указанной директории и ее подпапках,
    исключая __init__.py и __main__.py.
    """
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and file not in ["__init__.py", "__main__.py"]:
                py_files.append(os.path.join(root, file))
    return py_files

def combine_files_to_markdown(file_list, output_file):
    """
    Объединяет содержимое списка файлов в один Markdown-файл.
    """
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filepath in file_list:
            # Записываем заголовок с путем к файлу
            outfile.write(f"## {filepath}\\n\\n")
            outfile.write("```python\\n")
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    # Записываем содержимое файла
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Ошибка при чтении файла: {e}")
            outfile.write("\\n```\\n\\n")

if __name__ == "__main__":
    # Получаем список всех .py файлов
    all_py_files = get_all_py_files(TARGET_DIRECTORY)

    if all_py_files:
        # Объединяем файлы в один
        combine_files_to_markdown(all_py_files, OUTPUT_FILENAME)
        print(f"Все .py файлы были успешно объединены в '{OUTPUT_FILENAME}'")
    else:
        print("В указанной директории и ее подпапках не найдено .py файлов для объединения.")