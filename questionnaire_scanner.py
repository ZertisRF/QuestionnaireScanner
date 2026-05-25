from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
A4_ASPECT = 210 / 297


@dataclass(frozen=True)
class MarkerCandidate:
    center: tuple[float, float]
    radius: float
    area: float
    circularity: float
    fill_ratio: float
    paper_score: float
    background_gray: float
    background_saturation: float
    score: float


@dataclass(frozen=True)
class ScanConfig:
    max_side: int = 2200
    marker_margin_x: float = 0.09
    marker_margin_y: float = 0.055
    marker_box_width_mm: float = 170.0
    marker_box_height_mm: float = 260.0
    dark_threshold: int = 95
    crop_mode: str = "markers"
    output_width: int | None = None


@dataclass(frozen=True)
class ScanResult:
    source_path: Path
    output_path: Path
    debug_path: Path | None
    metadata_path: Path
    marker_points: np.ndarray
    document_points: np.ndarray
    output_size: tuple[int, int]


@dataclass(frozen=True)
class ScanBatchResult:
    source_path: Path
    out_dir: Path
    results: tuple[ScanResult, ...]
    failed: int
    total: int

    @property
    def processed(self) -> int:
        return len(self.results)


class ScanError(RuntimeError):
    pass


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ScanError(f"Не удалось прочитать изображение: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ScanError(f"Не удалось закодировать изображение: {path}")
    encoded.tofile(str(path))


def resize_for_processing(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image.copy(), 1.0

    scale = max_side / longest_side
    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError("Нужно ровно четыре точки формата (x, y)")

    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = pts.sum(axis=1)
    point_diff = pts[:, 1] - pts[:, 0]

    ordered[0] = pts[np.argmin(point_sum)]   # top-left
    ordered[2] = pts[np.argmax(point_sum)]   # bottom-right
    ordered[1] = pts[np.argmin(point_diff)]  # top-right
    ordered[3] = pts[np.argmax(point_diff)]  # bottom-left
    return ordered


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def build_dark_mask(gray: np.ndarray, dark_threshold: int) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, fixed = cv2.threshold(blurred, dark_threshold, 255, cv2.THRESH_BINARY_INV)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10,
    )
    mask = cv2.bitwise_and(fixed, adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def marker_background_score(
    gray: np.ndarray,
    saturation: np.ndarray,
    center: tuple[float, float],
    radius: float,
) -> tuple[float, float, float]:
    height, width = gray.shape[:2]
    x, y = center
    outer_radius = max(radius * 5.0, radius + 12)
    inner_radius = max(radius * 1.7, radius + 3)

    x_min = max(0, math.floor(x - outer_radius))
    x_max = min(width, math.ceil(x + outer_radius + 1))
    y_min = max(0, math.floor(y - outer_radius))
    y_max = min(height, math.ceil(y + outer_radius + 1))
    if x_max <= x_min or y_max <= y_min:
        return 0.0, 0.0, 255.0

    yy, xx = np.ogrid[y_min:y_max, x_min:x_max]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    ring = (distance >= inner_radius) & (distance <= outer_radius)
    if int(ring.sum()) < 20:
        return 0.0, 0.0, 255.0

    gray_values = gray[y_min:y_max, x_min:x_max][ring]
    saturation_values = saturation[y_min:y_max, x_min:x_max][ring]
    mean_gray = float(gray_values.mean())
    mean_saturation = float(saturation_values.mean())
    white_ratio = float(((gray_values > 125) & (saturation_values < 90)).mean())

    light_score = clamp((mean_gray - 80) / 120)
    low_saturation_score = clamp((120 - mean_saturation) / 120)
    paper_score = 0.45 * light_score + 0.35 * low_saturation_score + 0.20 * white_ratio
    return paper_score, mean_gray, mean_saturation


def detect_marker_candidates(image: np.ndarray, config: ScanConfig) -> list[MarkerCandidate]:
    resized, scale = resize_for_processing(image, config.max_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    mask = build_dark_mask(gray, config.dark_threshold)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    height, width = resized.shape[:2]
    min_side = min(height, width)
    min_radius = max(4.0, min_side * 0.003)
    max_radius = min_side * 0.028

    candidates: list[MarkerCandidate] = []
    hierarchy = hierarchy[0]
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= 0:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < min_radius or radius > max_radius:
            continue

        child_index = hierarchy[index][2]
        if child_index != -1:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(contour)
        aspect = w_box / max(h_box, 1)
        if not 0.35 <= aspect <= 2.85:
            continue

        circularity = 4 * math.pi * area / (perimeter * perimeter)
        circle_area = math.pi * radius * radius
        fill_ratio = area / circle_area
        if circularity < 0.35 or not 0.25 <= fill_ratio <= 1.18:
            continue

        paper_score, background_gray, background_saturation = marker_background_score(
            gray,
            saturation,
            (x, y),
            radius,
        )
        if paper_score < 0.28:
            continue

        contour_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        mean_gray = float(cv2.mean(gray, mask=contour_mask)[0])
        darkness = max(0.0, min(1.0, (120 - mean_gray) / 120))
        circle_quality = min(1.0, circularity) + (1 - min(abs(1.0 - fill_ratio), 1.0))
        score = circle_quality + darkness + 1.2 * paper_score

        candidates.append(
            MarkerCandidate(
                center=(x / scale, y / scale),
                radius=radius / scale,
                area=area / (scale * scale),
                circularity=circularity,
                fill_ratio=fill_ratio,
                paper_score=paper_score,
                background_gray=background_gray,
                background_saturation=background_saturation,
                score=score,
            )
        )

    return candidates


def select_corner_markers(
    candidates: list[MarkerCandidate],
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    if len(candidates) < 4:
        raise ScanError(f"Найдено только {len(candidates)} маркера, нужно 4")

    height, width = image_shape[:2]
    image_area = width * height
    center = np.array([width / 2, height / 2], dtype=np.float32)
    max_center_distance = float(np.linalg.norm(center))
    best_score = -1e9
    best_points: np.ndarray | None = None

    candidates_for_search = sorted(candidates, key=lambda item: item.score, reverse=True)[:40]
    for group in itertools.combinations(candidates_for_search, 4):
        points = order_points(np.array([item.center for item in group], dtype=np.float32))
        area = polygon_area(points)
        if area < image_area * 0.08:
            continue

        width_top = np.linalg.norm(points[1] - points[0])
        width_bottom = np.linalg.norm(points[2] - points[3])
        height_left = np.linalg.norm(points[3] - points[0])
        height_right = np.linalg.norm(points[2] - points[1])
        mean_width = (width_top + width_bottom) / 2
        mean_height = (height_left + height_right) / 2
        if mean_width <= 0 or mean_height <= 0:
            continue

        aspect = mean_width / mean_height
        if not 0.35 <= aspect <= 0.95:
            continue

        candidate_score = sum(item.score for item in group)
        distance_score = sum(np.linalg.norm(np.array(item.center) - center) for item in group)
        distance_score /= max_center_distance
        area_score = area / image_area
        paper_score = sum(item.paper_score for item in group)
        width_balance = min(width_top, width_bottom) / max(width_top, width_bottom)
        height_balance = min(height_left, height_right) / max(height_left, height_right)
        aspect_score = 1 - min(abs(aspect - A4_ASPECT), 0.6)
        score = (
            candidate_score
            + 3.0 * paper_score
            + 4.0 * area_score
            + 2.0 * width_balance
            + height_balance
            + 0.35 * distance_score
            + aspect_score
        )

        if score > best_score:
            best_score = score
            best_points = points

    if best_points is not None:
        return best_points

    raise ScanError("Не удалось выбрать согласованную четверку маркеров")


def document_corners_from_markers(marker_points: np.ndarray, config: ScanConfig) -> np.ndarray:
    margin_x = config.marker_margin_x
    margin_y = config.marker_margin_y
    if not 0 < margin_x < 0.45 or not 0 < margin_y < 0.45:
        raise ScanError("Отступы маркеров должны быть в диапазоне от 0 до 0.45")

    marker_template = np.array(
        [
            [margin_x, margin_y],
            [1 - margin_x, margin_y],
            [1 - margin_x, 1 - margin_y],
            [margin_x, 1 - margin_y],
        ],
        dtype=np.float32,
    )
    page_template = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(marker_template, marker_points.astype(np.float32))
    corners = cv2.perspectiveTransform(page_template.reshape(1, 4, 2), homography)
    return corners.reshape(4, 2).astype(np.float32)


def detect_paper_corners(image: np.ndarray, config: ScanConfig) -> np.ndarray | None:
    resized, scale = resize_for_processing(image, config.max_side)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    light_mask = ((saturation < 90) & (value > 105) & (gray > 95)).astype(np.uint8) * 255
    close_size = max(15, round(min(resized.shape[:2]) * 0.018))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    image_area = resized.shape[0] * resized.shape[1]
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image_area * 0.12:
        return None

    perimeter = cv2.arcLength(contour, True)
    approx = None
    for epsilon_factor in (0.015, 0.02, 0.03, 0.04, 0.055):
        candidate = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is None:
        rectangle = cv2.boxPoints(cv2.minAreaRect(contour))
        approx_points = rectangle.astype(np.float32)
    else:
        approx_points = approx.reshape(4, 2).astype(np.float32)

    return order_points(approx_points / scale)


def choose_document_corners(
    image: np.ndarray,
    marker_points: np.ndarray,
    config: ScanConfig,
) -> tuple[np.ndarray, str]:
    if config.crop_mode == "marker-box":
        return marker_points.astype(np.float32), "marker-box"

    marker_corners = document_corners_from_markers(marker_points, config)
    if config.crop_mode == "markers":
        return marker_corners, "markers"

    paper_corners = detect_paper_corners(image, config)
    if paper_corners is None:
        return marker_corners, "markers"

    marker_area = polygon_area(marker_points)
    paper_area = polygon_area(paper_corners)
    markers_inside = all(
        cv2.pointPolygonTest(paper_corners, tuple(point), measureDist=False) >= -1
        for point in marker_points
    )
    if config.crop_mode == "paper" or (markers_inside and paper_area > marker_area * 1.05):
        return paper_corners, "paper"

    return marker_corners, "markers"


def target_aspect(config: ScanConfig) -> float:
    if config.crop_mode == "marker-box":
        if config.marker_box_width_mm <= 0 or config.marker_box_height_mm <= 0:
            raise ScanError("Размер области между маркерами должен быть больше нуля")
        return config.marker_box_width_mm / config.marker_box_height_mm

    if not 0 < config.marker_margin_x < 0.45 or not 0 < config.marker_margin_y < 0.45:
        raise ScanError("Отступы маркеров должны быть в диапазоне от 0 до 0.45")

    return A4_ASPECT


def output_size_from_corners(
    corners: np.ndarray,
    output_width: int | None,
    aspect: float,
) -> tuple[int, int]:
    if output_width is not None:
        width = int(output_width)
        height = round(width / aspect)
        return width, height

    width_top = np.linalg.norm(corners[1] - corners[0])
    width_bottom = np.linalg.norm(corners[2] - corners[3])
    height_left = np.linalg.norm(corners[3] - corners[0])
    height_right = np.linalg.norm(corners[2] - corners[1])
    measured_width = max(width_top, width_bottom)
    measured_height = max(height_left, height_right)

    width_by_height = measured_height * aspect
    if width_by_height > measured_width:
        width = round(width_by_height)
        height = round(measured_height)
    else:
        width = round(measured_width)
        height = round(measured_width / aspect)

    return max(width, 1), max(height, 1)


def warp_document(image: np.ndarray, corners: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_debug(
    image: np.ndarray,
    candidates: list[MarkerCandidate],
    marker_points: np.ndarray,
    document_points: np.ndarray,
    corner_source: str,
) -> np.ndarray:
    debug = image.copy()

    for candidate in candidates:
        center = tuple(np.round(candidate.center).astype(int))
        cv2.circle(debug, center, max(2, round(candidate.radius)), (0, 180, 255), 2)

    labels = ("TL", "TR", "BR", "BL")
    for label, point in zip(labels, marker_points):
        center = tuple(np.round(point).astype(int))
        cv2.circle(debug, center, 22, (0, 0, 255), 5)
        cv2.putText(debug, label, (center[0] + 18, center[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    polygon = np.round(document_points).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(debug, [polygon], isClosed=True, color=(0, 255, 0), thickness=5)
    cv2.putText(
        debug,
        f"crop: {corner_source}",
        (35, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 120, 0),
        4,
    )
    return debug


def scan_image(
    source_path: Path,
    out_dir: Path,
    config: ScanConfig,
    debug: bool = False,
) -> ScanResult:
    validate_scan_config(config)
    image = read_image(source_path)
    candidates = detect_marker_candidates(image, config)
    marker_points = select_corner_markers(candidates, image.shape)
    document_points, corner_source = choose_document_corners(image, marker_points, config)
    output_size = output_size_from_corners(document_points, config.output_width, target_aspect(config))
    warped = warp_document(image, document_points, output_size)

    stem = source_path.stem
    output_path = out_dir / f"{stem}_scanned.png"
    debug_path = out_dir / f"{stem}_debug.png" if debug else None
    metadata_path = out_dir / f"{stem}_metadata.json"

    write_image(output_path, warped)
    if debug_path is not None:
        write_image(debug_path, draw_debug(image, candidates, marker_points, document_points, corner_source))

    metadata = {
        "source": str(source_path),
        "output": str(output_path),
        "debug": str(debug_path) if debug_path else None,
        "crop_source": corner_source,
        "output_size": {"width": output_size[0], "height": output_size[1]},
        "marker_points": marker_points.round(2).tolist(),
        "document_points": document_points.round(2).tolist(),
        "marker_candidates": [
            {
                "center": [round(candidate.center[0], 2), round(candidate.center[1], 2)],
                "radius": round(candidate.radius, 2),
                "area": round(candidate.area, 2),
                "paper_score": round(candidate.paper_score, 4),
                "background_gray": round(candidate.background_gray, 2),
                "background_saturation": round(candidate.background_saturation, 2),
                "score": round(candidate.score, 4),
            }
            for candidate in sorted(candidates, key=lambda item: item.score, reverse=True)
        ],
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return ScanResult(
        source_path=source_path,
        output_path=output_path,
        debug_path=debug_path,
        metadata_path=metadata_path,
        marker_points=marker_points,
        document_points=document_points,
        output_size=output_size,
    )


def validate_scan_config(config: ScanConfig) -> None:
    if config.crop_mode not in {"markers", "marker-box", "paper", "auto"}:
        raise ScanError(f"Неподдерживаемый режим обрезки: {config.crop_mode}")
    if config.max_side <= 0:
        raise ScanError("Максимальная сторона для обработки должна быть больше нуля")
    if not 0 <= config.dark_threshold <= 255:
        raise ScanError("Порог темных объектов должен быть в диапазоне от 0 до 255")
    if config.output_width is not None and config.output_width <= 0:
        raise ScanError("Ширина результата должна быть больше нуля")
    target_aspect(config)


def iter_images(source: Path) -> Iterable[Path]:
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ScanError(f"Неподдерживаемый формат файла: {source.suffix}")
        yield source
        return

    if not source.is_dir():
        raise ScanError(f"Путь не найден: {source}")

    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def scan_images(
    source: Path,
    out_dir: Path,
    config: ScanConfig,
    debug: bool = False,
    verbose: bool = True,
) -> ScanBatchResult:
    validate_scan_config(config)

    results: list[ScanResult] = []
    failed = 0
    total = 0

    for image_path in iter_images(source):
        total += 1
        try:
            result = scan_image(image_path, out_dir, config, debug=debug)
        except ScanError as error:
            failed += 1
            if verbose:
                print(f"FAIL: {image_path}: {error}")
            continue

        results.append(result)
        if verbose:
            print(f"OK: {image_path} -> {result.output_path} ({result.output_size[0]}x{result.output_size[1]})")

    if total == 0:
        raise ScanError(f"В папке нет изображений поддерживаемых форматов: {source}")

    if not results:
        raise ScanError(f"Не удалось обработать ни одно изображение из {source}; ошибок: {failed}")

    if failed and verbose:
        print(f"Готово: обработано {len(results)}, пропущено с ошибками {failed}")

    return ScanBatchResult(
        source_path=source,
        out_dir=out_dir,
        results=tuple(results),
        failed=failed,
        total=total,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Поиск маркеров анкеты и перспективное выравнивание фотографии.",
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=Path("input"),
        help="Путь к изображению или папке с изображениями. По умолчанию: input",
    )
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("output"), help="Папка для результата")
    parser.add_argument("--debug", action="store_true", help="Сохранить изображение с найденными маркерами")
    parser.add_argument("--max-side", type=int, default=2200, help="Максимальная сторона для поиска маркеров")
    parser.add_argument("--dark-threshold", type=int, default=95, help="Порог темных объектов от 0 до 255")
    parser.add_argument("--margin-x", type=float, default=0.09, help="Горизонтальный отступ маркеров от края листа")
    parser.add_argument("--margin-y", type=float, default=0.055, help="Вертикальный отступ маркеров от края листа")
    parser.add_argument("--marker-box-width-mm", type=float, default=170.0, help="Расстояние между верхними маркерами в мм")
    parser.add_argument("--marker-box-height-mm", type=float, default=260.0, help="Расстояние между левыми маркерами в мм")
    parser.add_argument(
        "--crop-mode",
        choices=("markers", "marker-box", "paper", "auto"),
        default="markers",
        help="Источник углов: полный лист по маркерам, область между маркерами, контур бумаги или авто-выбор",
    )
    parser.add_argument("--output-width", type=int, default=None, help="Фиксированная ширина результата в пикселях")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ScanConfig(
        max_side=args.max_side,
        marker_margin_x=args.margin_x,
        marker_margin_y=args.margin_y,
        marker_box_width_mm=args.marker_box_width_mm,
        marker_box_height_mm=args.marker_box_height_mm,
        dark_threshold=args.dark_threshold,
        crop_mode=args.crop_mode,
        output_width=args.output_width,
    )
    scan_images(args.source, args.out_dir, config, debug=args.debug)


if __name__ == "__main__":
    main()
