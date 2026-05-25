# QuestionnaireScanner

Проект обрабатывает фотографии анкет: находит четыре черных маркера, исправляет перспективу, обрезает анкету и размечает клетки ответов по JSON-шаблону.

## Что Важно

Файлы проекта:

- `Main.py` — основной запуск, в том числе полный pipeline.
- `questionnaire_scanner.py` — поиск маркеров, обрезка, перспективное выравнивание.
- `crop_by_markers.py` — быстрый запуск обрезки по центрам маркеров.
- `mark_answer_cells.py` — разметка клеток красными рамками и сохранение координат.
- `questionnaire_layout.json` — пример шаблона анкеты в миллиметрах.
- `requirements.txt` — зависимости.

Рабочие данные:

- `input/` — входные фотографии анкет.
- `output/` — выровненный полный лист.
- `output_marker_crop/` — анкета, обрезанная по маркерам.
- `output_answer_markup/` — анкета с красными рамками клеток и JSON-координатами.
- `output*_test/` и любые другие `output*` — проверочные/сгенерированные результаты.

Папки `output*` можно удалить: они не являются исходным кодом и создаются заново при запуске. Они уже добавлены в `.gitignore`.

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Полный Запуск

Положите фотографии в `input/`, затем выполните:

```bash
.venv/bin/python Main.py --all --debug
```

Команда последовательно выполнит три этапа:

1. `input -> output`: выравнивание полного листа.
2. `input -> output_marker_crop`: обрезка по центрам маркеров.
3. `output_marker_crop -> output_answer_markup`: разметка клеток ответов.

Флаг `--debug` сохраняет изображения с найденными маркерами и рамками.

## Отдельные Этапы

Обычное выравнивание полного листа:

```bash
.venv/bin/python Main.py --debug
```

Обрезка только по маркерам:

```bash
.venv/bin/python crop_by_markers.py --debug
```

Разметка клеток на уже обрезанных анкетах:

```bash
.venv/bin/python mark_answer_cells.py
```

## Разметка Клеток

Пример шаблона находится в `questionnaire_layout.json`.

Сейчас в нем задано:

- расстояние между верхними маркерами: `170 мм`;
- расстояние между левыми маркерами: `260 мм`;
- начало таблиц: `25 мм` вправо от левого верхнего маркера;
- первый разделитель двухъярусной таблицы: `55 мм` вниз;
- шаг таблиц по вертикали: `25 мм`;
- размер клетки: `10x10 мм`;
- 4 таблицы, 2 строки, 11 колонок.

`crop_by_markers.py` сохраняет изображение `1700x2600 px`, поэтому масштаб получается ровным: `10 px = 1 мм`, а клетка `10x10 мм` становится `100x100 px`.

Для каждой анкеты `mark_answer_cells.py` создает:

- `*_cells_marked.png` — изображение с красными рамками;
- `*_cells.json` — координаты каждой клетки в `rect_mm` и `rect_px`.

По умолчанию рамки слегка подгоняются к реально найденным линиям таблицы. Если нужно использовать только координаты из JSON-шаблона, запустите:

```bash
.venv/bin/python mark_answer_cells.py --no-snap-to-grid
```

По умолчанию используется более устойчивый режим `--snap-mode uniform`: по горизонтали он ищет внешние границы каждой таблицы и делит ее на ровные колонки, а по вертикали отдельно проверяет верхнюю линию, разделитель строк и нижнюю линию. Это помогает, когда на фото штрихи букв похожи на внутренние линии сетки. Старый режим, где каждая линия ищется отдельно, можно включить так:

```bash
.venv/bin/python mark_answer_cells.py --snap-mode individual
```

## Настройка

Если маркеры в печатном шаблоне стоят иначе, можно изменить отступы полного листа:

```bash
.venv/bin/python Main.py photo.jpg --margin-x 0.08 --margin-y 0.06 --debug
```

Если нужно поменять расстояние между маркерами для marker-box режима:

```bash
.venv/bin/python Main.py input -o output_marker_crop --crop-mode marker-box --marker-box-width-mm 170 --marker-box-height-mm 260 --output-width 1700
```

Если нужен более легкий датасет, можно уменьшить ширину, например до `850`; высота будет рассчитана автоматически:

```bash
.venv/bin/python Main.py input -o output_marker_crop --crop-mode marker-box --output-width 850
```

## Быстрая Проверка

```bash
.venv/bin/python -m py_compile Main.py questionnaire_scanner.py crop_by_markers.py mark_answer_cells.py
.venv/bin/python Main.py --all --debug
```
# QuestionnaireScanner
# QuestionnaireScanner
# QuestionnaireScanner
