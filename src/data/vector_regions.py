"""Deterministic geometric grouping for PDF vector drawing diagnostics.

Groups are candidate visual regions, not figure classifications. Coordinates
are PyMuPDF page coordinates in points, with origin at the displayed page's
top-left; input boxes are clipped to the page before grouping. The default
connection distance is 0.006 of the shorter page edge (about 3.7 pt on A4),
which only joins touching or very closely spaced geometry. This deliberately
conservative starting point is configurable and must not be treated as a
semantic figure detector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DrawingBox:
    drawing_index: int
    bbox: tuple[float, float, float, float]


def _values(rect: Any) -> tuple[float, float, float, float]:
    try:
        result = tuple(float(v) for v in rect)
    except (TypeError, ValueError) as exc:
        raise ValueError("drawing rect must contain four numeric coordinates") from exc
    if len(result) != 4 or not all(math.isfinite(v) for v in result):
        raise ValueError("drawing rect coordinates must be four finite numbers")
    x0, y0, x1, y1 = result
    if x1 <= x0 or y1 <= y0:
        raise ValueError("drawing rect must have positive width and height")
    return x0, y0, x1, y1


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def group_drawing_regions(
    drawings: Iterable[Mapping[str, Any]],
    *,
    page_width: float,
    page_height: float,
    distance_fraction: float = 0.006,
) -> dict[str, Any]:
    """Group intersecting/nearby drawing rectangles with a deterministic sweep.

    The original sequence position is retained as ``drawing_index``. Invalid
    boxes are reported in ``rejected_drawings`` instead of silently discarded.
    Out-of-page boxes are clipped to ``[0,width] × [0,height]``; boxes with no
    visible area after clipping are reported as rejected. Distance is the
    Euclidean gap between rectangle edges in page points, not center distance.
    """
    width, height, fraction = float(page_width), float(page_height), float(distance_fraction)
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
        raise ValueError("page_width and page_height must be positive finite point values")
    if not math.isfinite(fraction) or fraction < 0:
        raise ValueError("distance_fraction must be a finite non-negative value")
    threshold = min(width, height) * fraction

    valid: list[DrawingBox] = []
    rejected: list[dict[str, Any]] = []
    for index, drawing in enumerate(drawings):
        try:
            raw = _values(drawing.get("rect"))
            clipped = (
                max(0.0, raw[0]),
                max(0.0, raw[1]),
                min(width, raw[2]),
                min(height, raw[3]),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                raise ValueError("drawing rectangle has no visible area inside the page")
            valid.append(DrawingBox(index, clipped))
        except (TypeError, ValueError) as exc:
            rejected.append({"drawing_index": index, "reason": str(exc)})

    parent = list(range(len(valid)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    # Sweep by x0. Expired boxes cannot overlap or be within threshold of the
    # current box; remaining candidates are checked for full 2D edge distance.
    ordered = sorted(range(len(valid)), key=lambda i: (valid[i].bbox[0], valid[i].bbox[1], valid[i].drawing_index))
    active: list[int] = []
    for current in ordered:
        box = valid[current].bbox
        active = [i for i in active if valid[i].bbox[2] + threshold >= box[0]]
        for other in active:
            if _distance(box, valid[other].bbox) <= threshold:
                union(current, other)
        active.append(current)

    groups: dict[int, list[DrawingBox]] = {}
    for index, box in enumerate(valid):
        groups.setdefault(find(index), []).append(box)

    regions = []
    for members in groups.values():
        members.sort(key=lambda item: item.drawing_index)
        bounds = (
            min(item.bbox[0] for item in members),
            min(item.bbox[1] for item in members),
            max(item.bbox[2] for item in members),
            max(item.bbox[3] for item in members),
        )
        regions.append({
            "bbox": {"x0": bounds[0], "y0": bounds[1], "x1": bounds[2], "y1": bounds[3]},
            "drawing_indices": [item.drawing_index for item in members],
            "object_count": len(members),
        })
    regions.sort(key=lambda region: (region["bbox"]["y0"], region["bbox"]["x0"], region["drawing_indices"][0]))
    return {
        "coordinate_system": "PyMuPDF page points; top-left origin; boxes clipped to page bounds",
        "page_size": {"width": width, "height": height},
        "distance_threshold_points": threshold,
        "distance_fraction_of_short_edge": fraction,
        "candidate_region_count": len(regions),
        "regions": regions,
        "rejected_drawings": rejected,
    }
