# image-convolution-practice
Репозиторий для выполнения учебной практики по теме свертка изображений.

## Структура проекта
* `src/kernels.py` — библиотека матриц ядер свертки.
* `src/convolution.py` — класс `ImageConvolution` (математика паддинга и свертки).
* `src/main.py` — точка входа, консольный интерфейс (CLI).
* `tests/test_convolution.py` — интеграционные Golden-тесты.
* `data/` — папка для демонстрационных изображений.

## Использование (CLI)

Запуск утилиты выполняется из корня проекта через вызов модуля:

```bash
python3 -m src.main data/input.jpg --kernel sharpen --mode reflect --out data/result.png
```

### Параметры:
* `input` — путь к исходному изображению (обязательный аргумент).
* `--kernel` — выбор фильтра (`sharpen`, `blur`, `sobel`). По умолчанию: `sharpen`.
* `--mode` — режим обработки краев (`zero`, `edge`, `reflect`). По умолчанию: `zero`.
* `--out` — путь для сохранения результата. По умолчанию: `result.png`.

## Тестирование и автоматические проверки (CI)

### 1. Запуск интеграционных Golden-тестов
Тесты автоматически проверяют все комбинации ядер и режимов паддинга.

```bash
PYTHONPATH=src pytest tests/test_convolution.py -v
```

### 2. Проверка стиля кода и форматирования (Ruff)
```bash
ruff check src/ --ignore=E501,E203
```

### 3. Статическая проверка типов (Mypy)
```bash
MYPYPATH=src mypy src/ --ignore-missing-imports
```
## Источники
* Исходное демонстрационное изображение (`data/input.jpg`): [https://www.fotoprizer.ru/img/221579_big.jpg]
