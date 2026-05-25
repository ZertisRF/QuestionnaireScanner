from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from questionnaire_scanner import SUPPORTED_EXTENSIONS, ScanError, read_image, write_image


GENERATED_IMAGE_SUFFIXES = ("_debug", "_cells_marked")
SNAP_MODES = {"uniform", "individual"}


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ScanError(f"Неподдерживаемый формат файла: {source.suffix}")
        yield source
        return

    if not source.is_dir():
        raise ScanError(f"Путь не найден: {source}")

    for path in sorted(source.iterdir()):
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.stem.endswith(GENERATED_IMAGE_SUFFIXES)
        ):
            yield path


def load_layout(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        layout = json.load(file)

    marker_box = layout.get("marker_box_mm", {})
    grid = layout.get("answer_grid", {})
    cell = grid.get("cell_mm", {})
    rows = grid.get("rows")
    required = [
        marker_box.get("width"),
        marker_box.get("height"),
        grid.get("left_mm"),
        grid.get("first_separator_y_mm"),
        grid.get("table_step_y_mm"),
        grid.get("tables"),
        grid.get("columns"),
        cell.get("width"),
        cell.get("height"),
        rows,
    ]
    if any(value is None for value in required):
        raise ScanError(f"В layout-файле не хватает параметров: {path}")
    if not isinstance(rows, list) or not rows:
        raise ScanError(f"В layout-файле должен быть непустой список строк answer_grid.rows: {path}")

    positive_values = {
        "marker_box_mm.width": marker_box["width"],
        "marker_box_mm.height": marker_box["height"],
        "answer_grid.tables": grid["tables"],
        "answer_grid.columns": grid["columns"],
        "answer_grid.cell_mm.width": cell["width"],
        "answer_grid.cell_mm.height": cell["height"],
    }
    for name, value in positive_values.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ScanError(f"Параметр {name} должен быть числом") from error
        if numeric_value <= 0:
            raise ScanError(f"Параметр {name} должен быть больше нуля")

    for index, row in enumerate(rows, start=1):
        if "id" not in row or "y_offset_from_separator_mm" not in row:
            raise ScanError(f"В строке answer_grid.rows[{index}] не хватает id или y_offset_from_separator_mm")

    return layout


def build_cells(layout: dict) -> list[dict]:
    grid = layout["answer_grid"]
    cell_width = float(grid["cell_mm"]["width"])
    cell_height = float(grid["cell_mm"]["height"])
    cells: list[dict] = []

    for table_index in range(int(grid["tables"])):
        separator_y = float(grid["first_separator_y_mm"]) + table_index * float(grid["table_step_y_mm"])
        for row_index, row in enumerate(grid["rows"]):
            y_mm = separator_y + float(row["y_offset_from_separator_mm"])
            for column_index in range(int(grid["columns"])):
                x_mm = float(grid["left_mm"]) + column_index * cell_width
                cells.append(
                    {
                        "id": f"table_{table_index + 1:02d}_{row['id']}_col_{column_index + 1:02d}",
                        "table_index": table_index + 1,
                        "row_index": row_index + 1,
                        "row_type": row["id"],
                        "row_label": row.get("label", row["id"]),
                        "column_index": column_index + 1,
                        "rect_mm": {
                            "x": round(x_mm, 3),
                            "y": round(y_mm, 3),
                            "width": round(cell_width, 3),
                            "height": round(cell_height, 3),
                        },
                    }
                )

    return cells


def mm_rect_to_px(rect_mm: dict, px_per_mm_x: float, px_per_mm_y: float) -> dict:
    x = round(float(rect_mm["x"]) * px_per_mm_x)
    y = round(float(rect_mm["y"]) * px_per_mm_y)
    width = max(1, round(float(rect_mm["width"]) * px_per_mm_x))
    height = max(1, round(float(rect_mm["height"]) * px_per_mm_y))
    return {"x": x, "y": y, "width": width, "height": height}


def smooth_profile(profile: np.ndarray, window: int = 5) -> np.ndarray:
    if profile.size < window:
        return profile
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(profile.astype(np.float32), kernel, mode="same")


def build_grid_line_masks(image: np.ndarray, px_per_mm_x: float, px_per_mm_y: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        12,
    )

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(21, round(px_per_mm_y * 4))))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(21, round(px_per_mm_x * 4)), 1))
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    horizontal_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    return vertical_mask, horizontal_mask


def find_best_line(
    mask: np.ndarray,
    expected: int,
    search_radius: int,
    axis: str,
    span_start: int,
    span_end: int,
) -> int:
    height, width = mask.shape[:2]
    if axis == "x":
        left = max(0, expected - search_radius)
        right = min(width, expected + search_radius + 1)
        top = max(0, span_start)
        bottom = min(height, span_end)
        if right <= left or bottom <= top:
            return expected
        profile = mask[top:bottom, left:right].sum(axis=0)
        smoothed = smooth_profile(profile)
        if smoothed.size == 0 or float(smoothed.max()) <= 0:
            return expected
        return left + int(np.argmax(smoothed))

    if axis != "y":
        raise ValueError(f"Неподдерживаемая ось поиска линии: {axis}")

    top = max(0, expected - search_radius)
    bottom = min(height, expected + search_radius + 1)
    left = max(0, span_start)
    right = min(width, span_end)
    if right <= left or bottom <= top:
        return expected
    profile = mask[top:bottom, left:right].sum(axis=1)
    smoothed = smooth_profile(profile)
    if smoothed.size == 0 or float(smoothed.max()) <= 0:
        return expected
    return top + int(np.argmax(smoothed))


def has_strictly_increasing_steps(values: list[int]) -> bool:
    return all(values[index + 1] > values[index] for index in range(len(values) - 1))


def line_spacing_ratios(lines: list[int], expected_lines: list[int]) -> list[float]:
    ratios: list[float] = []
    for index in range(len(lines) - 1):
        expected_step = expected_lines[index + 1] - expected_lines[index]
        snapped_step = lines[index + 1] - lines[index]
        if expected_step <= 0:
            return []
        ratios.append(snapped_step / expected_step)
    return ratios


def spacing_is_reasonable(
    lines: list[int],
    expected_lines: list[int],
    min_step_ratio: float,
    max_step_ratio: float,
) -> bool:
    if not has_strictly_increasing_steps(lines):
        return False
    ratios = line_spacing_ratios(lines, expected_lines)
    return bool(ratios) and all(min_step_ratio <= ratio <= max_step_ratio for ratio in ratios)


def evenly_spaced_lines(first: int, last: int, count: int) -> list[int]:
    if count <= 1:
        return [int(first)]
    return [int(round(value)) for value in np.linspace(first, last, count)]


def snap_uniform_lines(
    mask: np.ndarray,
    expected_lines: list[int],
    search_radius_px: int,
    axis: str,
    span_start: int,
    span_end: int,
    min_span_ratio: float = 0.75,
    max_span_ratio: float = 1.25,
) -> tuple[list[int], dict]:
    if len(expected_lines) < 2:
        return expected_lines, {"source": "layout"}

    first = find_best_line(mask, expected_lines[0], search_radius_px, axis, span_start, span_end)
    last = find_best_line(mask, expected_lines[-1], search_radius_px, axis, span_start, span_end)
    expected_span = expected_lines[-1] - expected_lines[0]
    snapped_span = last - first
    details = {
        "source": "uniform_edges",
        "first": first,
        "last": last,
        "expected_span": expected_span,
        "snapped_span": snapped_span,
    }

    if expected_span <= 0 or snapped_span <= 0:
        return expected_lines, {**details, "fallback": "non_positive_span"}

    span_ratio = snapped_span / expected_span
    details["span_ratio"] = round(span_ratio, 4)
    if not min_span_ratio <= span_ratio <= max_span_ratio:
        return expected_lines, {**details, "fallback": "span_ratio_out_of_range"}

    lines = evenly_spaced_lines(first, last, len(expected_lines))
    if not has_strictly_increasing_steps(lines):
        return expected_lines, {**details, "fallback": "non_increasing_lines"}

    return lines, details


def snap_individual_lines(
    mask: np.ndarray,
    expected_lines: list[int],
    search_radius_px: int,
    axis: str,
    span_start: int,
    span_end: int,
    min_step_ratio: float = 0.75,
    max_step_ratio: float = 1.25,
) -> tuple[list[int], dict]:
    lines = [
        find_best_line(mask, expected, search_radius_px, axis, span_start, span_end)
        for expected in expected_lines
    ]
    if not spacing_is_reasonable(lines, expected_lines, min_step_ratio, max_step_ratio):
        return expected_lines, {"source": "layout", "fallback": "invalid_spacing", "snapped_lines": lines}

    return lines, {"source": "individual_lines", "step_ratios": [round(ratio, 4) for ratio in line_spacing_ratios(lines, expected_lines)]}


def snap_cells_to_printed_grid(
    image: np.ndarray,
    cells: list[dict],
    px_per_mm_x: float,
    px_per_mm_y: float,
    search_radius_px: int,
    snap_mode: str,
) -> tuple[list[dict], dict]:
    if snap_mode not in SNAP_MODES:
        raise ScanError(f"Неподдерживаемый режим привязки сетки: {snap_mode}")

    vertical_mask, horizontal_mask = build_grid_line_masks(image, px_per_mm_x, px_per_mm_y)
    snapped_cells = [dict(cell) for cell in cells]
    snap_tables: dict[str, dict] = {}

    table_indexes = sorted({cell["table_index"] for cell in cells})
    for table_index in table_indexes:
        table_cells = [cell for cell in cells if cell["table_index"] == table_index]
        printed_cells = [cell for cell in table_cells if cell["row_type"] == "printed"]
        answer_cells = [cell for cell in table_cells if cell["row_type"] == "answer"]
        if not printed_cells or not answer_cells:
            continue

        first_printed = min(printed_cells, key=lambda item: item["column_index"])
        first_answer = min(answer_cells, key=lambda item: item["column_index"])
        cell_width_px = first_printed["rect_px"]["width"]
        expected_x = [
            first_printed["rect_px"]["x"] + column_index * cell_width_px
            for column_index in range(len(printed_cells) + 1)
        ]
        expected_y = [
            first_printed["rect_px"]["y"],
            first_answer["rect_px"]["y"],
            first_answer["rect_px"]["y"] + first_answer["rect_px"]["height"],
        ]

        x_span_start = expected_x[0] - search_radius_px
        x_span_end = expected_x[-1] + search_radius_px
        y_span_start = expected_y[0] - search_radius_px
        y_span_end = expected_y[-1] + search_radius_px
        cell_height_px = first_answer["rect_px"]["height"]
        x_edge_search_radius_px = max(search_radius_px, round(cell_width_px * 0.45))
        y_edge_search_radius_px = min(search_radius_px, max(12, round(cell_height_px * 0.25)))

        if snap_mode == "uniform":
            snapped_x, x_snap_details = snap_uniform_lines(
                vertical_mask,
                expected_x,
                x_edge_search_radius_px,
                "x",
                y_span_start,
                y_span_end,
            )
            snapped_y, y_snap_details = snap_uniform_lines(
                horizontal_mask,
                expected_y,
                y_edge_search_radius_px,
                "y",
                snapped_x[0] - search_radius_px,
                snapped_x[-1] + search_radius_px,
                min_span_ratio=0.88,
                max_span_ratio=1.12,
            )
            individual_y, individual_y_details = snap_individual_lines(
                horizontal_mask,
                expected_y,
                y_edge_search_radius_px,
                "y",
                snapped_x[0] - search_radius_px,
                snapped_x[-1] + search_radius_px,
                min_step_ratio=0.88,
                max_step_ratio=1.12,
            )
            y_snap_details = {
                "edge_snap": y_snap_details,
                "line_snap": individual_y_details,
            }
            if individual_y_details.get("source") == "individual_lines":
                snapped_y = individual_y
                y_snap_details["source"] = "individual_horizontal_lines"
            else:
                y_snap_details["source"] = "uniform_edges"
        else:
            snapped_x, x_snap_details = snap_individual_lines(
                vertical_mask,
                expected_x,
                search_radius_px,
                "x",
                y_span_start,
                y_span_end,
            )
            snapped_y, y_snap_details = snap_individual_lines(
                horizontal_mask,
                expected_y,
                search_radius_px,
                "y",
                x_span_start,
                x_span_end,
            )

        snap_tables[str(table_index)] = {
            "snap_mode": snap_mode,
            "expected_x": expected_x,
            "expected_y": expected_y,
            "snapped_x": snapped_x,
            "snapped_y": snapped_y,
            "x_snap": x_snap_details,
            "y_snap": y_snap_details,
        }

        for cell in snapped_cells:
            if cell["table_index"] != table_index:
                continue
            column = cell["column_index"] - 1
            row = 0 if cell["row_type"] == "printed" else 1
            cell["rect_px"] = {
                "x": int(snapped_x[column]),
                "y": int(snapped_y[row]),
                "width": int(snapped_x[column + 1] - snapped_x[column]),
                "height": int(snapped_y[row + 1] - snapped_y[row]),
            }
            cell["rect_px_source"] = f"snapped_grid_{snap_mode}"

    return snapped_cells, snap_tables


def draw_cells(
    image_path: Path,
    out_dir: Path,
    layout_path: Path,
    layout: dict,
    thickness: int,
    snap_to_grid: bool,
    snap_search_radius_px: int,
    snap_mode: str = "uniform",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)
    height_px, width_px = image.shape[:2]
    marker_width_mm = float(layout["marker_box_mm"]["width"])
    marker_height_mm = float(layout["marker_box_mm"]["height"])
    px_per_mm_x = width_px / marker_width_mm
    px_per_mm_y = height_px / marker_height_mm

    cells = []
    for cell in build_cells(layout):
        rect_px = mm_rect_to_px(cell["rect_mm"], px_per_mm_x, px_per_mm_y)
        cells.append(
            {
                **cell,
                "rect_px": rect_px,
                "rect_px_source": "layout",
            }
        )

    snap_tables = {}
    if snap_to_grid:
        cells, snap_tables = snap_cells_to_printed_grid(
            image,
            cells,
            px_per_mm_x,
            px_per_mm_y,
            snap_search_radius_px,
            snap_mode,
        )

    marked = image.copy()
    for cell in cells:
        rect_px = cell["rect_px"]
        x1 = max(0, min(width_px - 1, rect_px["x"]))
        y1 = max(0, min(height_px - 1, rect_px["y"]))
        x2 = max(0, min(width_px - 1, rect_px["x"] + rect_px["width"] - 1))
        y2 = max(0, min(height_px - 1, rect_px["y"] + rect_px["height"] - 1))
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), thickness)

    stem = image_path.stem.removesuffix("_scanned")
    marked_path = out_dir / f"{stem}_cells_marked.png"
    cells_path = out_dir / f"{stem}_cells.json"
    write_image(marked_path, marked)

    payload = {
        "source_image": str(image_path),
        "marked_image": str(marked_path),
        "layout_file": str(layout_path),
        "layout_name": layout.get("name"),
        "coordinate_system": layout.get("coordinate_system"),
        "marker_box_mm": layout["marker_box_mm"],
        "image_size_px": {"width": width_px, "height": height_px},
        "px_per_mm": {"x": px_per_mm_x, "y": px_per_mm_y},
        "snap_to_grid": snap_to_grid,
        "snap_mode": snap_mode if snap_to_grid else None,
        "snap_search_radius_px": snap_search_radius_px if snap_to_grid else None,
        "snap_tables": snap_tables,
        "cells_count": len(cells),
        "cells": cells,
    }
    cells_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return marked_path, cells_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Нанести красные рамки ячеек ответов на выровненные marker-box анкеты.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=Path("output_marker_crop"),
        help="Файл или папка с анкетами, обрезанными по маркерам. По умолчанию: output_marker_crop",
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("output_answer_markup"), help="Папка результата")
    parser.add_argument("--layout", type=Path, default=Path("questionnaire_layout.json"), help="JSON-шаблон анкеты")
    parser.add_argument("--thickness", type=int, default=2, help="Толщина красной рамки в пикселях")
    parser.add_argument("--no-snap-to-grid", action="store_true", help="Не подгонять рамки к найденным линиям таблицы")
    parser.add_argument("--snap-search-radius", type=int, default=25, help="Радиус поиска линий таблицы в пикселях")
    parser.add_argument(
        "--snap-mode",
        choices=sorted(SNAP_MODES),
        default="uniform",
        help="uniform: внешние границы и ровный шаг; individual: искать каждую линию отдельно",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.thickness <= 0:
        raise ScanError("Толщина рамки должна быть больше нуля")
    if args.snap_search_radius < 0:
        raise ScanError("Радиус поиска линий таблицы не может быть отрицательным")

    layout = load_layout(args.layout)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for image_path in iter_images(args.source):
        marked_path, cells_path = draw_cells(
            image_path,
            args.out_dir,
            args.layout,
            layout,
            args.thickness,
            snap_to_grid=not args.no_snap_to_grid,
            snap_search_radius_px=args.snap_search_radius,
            snap_mode=args.snap_mode,
        )
        processed += 1
        print(f"OK: {image_path} -> {marked_path}, {cells_path}")

    if processed == 0:
        raise ScanError(f"Нет изображений для разметки: {args.source}")


if __name__ == "__main__":
    main()
