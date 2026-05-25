from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from mark_answer_cells import SNAP_MODES, draw_cells, load_layout
from questionnaire_scanner import (
    ScanConfig,
    ScanError,
    main as scanner_main,
    parse_args as parse_scanner_args,
    scan_images,
)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Запуск сканера анкеты или полного pipeline: scan -> crop -> answer markup.",
        add_help=False,
    )
    parser.add_argument("--all", "--pipeline", action="store_true", dest="pipeline", help="Выполнить все этапы подряд")
    parser.add_argument(
        "--crop-out-dir",
        type=Path,
        default=Path("output_marker_crop"),
        help="Папка для marker-box обрезки в полном pipeline",
    )
    parser.add_argument(
        "--crop-output-width",
        type=int,
        default=1700,
        help="Ширина marker-box результата в пикселях",
    )
    parser.add_argument(
        "--markup-out-dir",
        type=Path,
        default=Path("output_answer_markup"),
        help="Папка для разметки клеток в полном pipeline",
    )
    parser.add_argument("--layout", type=Path, default=Path("questionnaire_layout.json"), help="JSON-шаблон анкеты")
    parser.add_argument("--thickness", type=int, default=2, help="Толщина рамки клеток в пикселях")
    parser.add_argument("--no-snap-to-grid", action="store_true", help="Не подгонять рамки к найденным линиям таблицы")
    parser.add_argument("--snap-search-radius", type=int, default=25, help="Радиус поиска линий таблицы в пикселях")
    parser.add_argument(
        "--snap-mode",
        choices=sorted(SNAP_MODES),
        default="uniform",
        help="uniform: внешние границы и ровный шаг; individual: искать каждую линию отдельно",
    )
    return parser.parse_known_args(argv)


def build_scan_config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        max_side=args.max_side,
        marker_margin_x=args.margin_x,
        marker_margin_y=args.margin_y,
        marker_box_width_mm=args.marker_box_width_mm,
        marker_box_height_mm=args.marker_box_height_mm,
        dark_threshold=args.dark_threshold,
        crop_mode=args.crop_mode,
        output_width=args.output_width,
    )


def run_pipeline(argv: list[str] | None = None) -> None:
    pipeline_args, scanner_argv = parse_args(argv)
    scanner_args = parse_scanner_args(scanner_argv)
    scan_config = build_scan_config(scanner_args)
    if pipeline_args.crop_output_width <= 0:
        raise ScanError("Ширина marker-box результата должна быть больше нуля")
    if pipeline_args.thickness <= 0:
        raise ScanError("Толщина рамки должна быть больше нуля")
    if pipeline_args.snap_search_radius < 0:
        raise ScanError("Радиус поиска линий таблицы не может быть отрицательным")

    print("Шаг 1/3: полное выравнивание анкеты")
    scan_images(scanner_args.source, scanner_args.out_dir, scan_config, debug=scanner_args.debug)

    print("Шаг 2/3: обрезка по маркерам")
    crop_config = replace(scan_config, crop_mode="marker-box", output_width=pipeline_args.crop_output_width)
    crop_batch = scan_images(scanner_args.source, pipeline_args.crop_out_dir, crop_config, debug=scanner_args.debug)

    print("Шаг 3/3: разметка клеток ответов")
    layout = load_layout(pipeline_args.layout)
    for crop_result in crop_batch.results:
        marked_path, cells_path = draw_cells(
            crop_result.output_path,
            pipeline_args.markup_out_dir,
            pipeline_args.layout,
            layout,
            pipeline_args.thickness,
            snap_to_grid=not pipeline_args.no_snap_to_grid,
            snap_search_radius_px=pipeline_args.snap_search_radius,
            snap_mode=pipeline_args.snap_mode,
        )
        print(f"OK: {crop_result.output_path} -> {marked_path}, {cells_path}")

    print(
        "Готово: "
        f"scan={scanner_args.out_dir}, crop={pipeline_args.crop_out_dir}, markup={pipeline_args.markup_out_dir}"
    )


def main(argv: list[str] | None = None) -> None:
    pipeline_args, scanner_argv = parse_args(argv)
    if not pipeline_args.pipeline:
        scanner_main(scanner_argv)
        return

    run_pipeline(argv)


if __name__ == "__main__":
    main()
