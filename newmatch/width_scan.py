import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip

PX_PER_INCH = 100


def measure_widths(mask, base, tip, step_px=PX_PER_INCH):
    """Walk from base to tip along the base->tip line in step_px increments.
    At each sample point, measure the leaf width along the perpendicular to
    the base->tip direction. Returns sample centers, perpendicular unit
    vector, and the measured widths (in px)."""

    base = np.array(base, dtype=np.float64)
    tip = np.array(tip, dtype=np.float64)
    direction = tip - base
    length = np.linalg.norm(direction)
    direction /= length

    perp = np.array([-direction[1], direction[0]])

    n_steps = int(length // step_px) + 1
    centers = []
    widths = []
    edges = []

    h, w = mask.shape
    max_half = int(np.hypot(h, w))

    for i in range(n_steps + 1):
        d = min(i * step_px, length)
        center = base + direction * d

        # sample mask values along the full perpendicular line, from
        # -max_half to +max_half, and take the outermost hits as the edges
        # (robust to the center point itself falling outside the mask,
        # e.g. over a region that's been eaten away)
        t_range = np.arange(-max_half, max_half + 1)
        xs = np.round(center[0] + perp[0] * t_range).astype(int)
        ys = np.round(center[1] + perp[1] * t_range).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hit = np.zeros_like(valid)
        hit[valid] = mask[ys[valid], xs[valid]] > 0

        hit_idx = np.where(hit)[0]
        if len(hit_idx) == 0:
            width = 0
            edge_lo, edge_hi = 0, 0
        else:
            edge_lo, edge_hi = t_range[hit_idx[0]], t_range[hit_idx[-1]]
            width = edge_hi - edge_lo

        centers.append(center)
        widths.append(width)
        edges.append((edge_lo, edge_hi))

        if d >= length:
            break

    return np.array(centers), perp, np.array(widths), edges


def delta_color(delta_inch):
    """Green for no change/growth, red for a sharp decrease in width.
    Scales with magnitude of decrease, capped at 1 inch drop."""
    if delta_inch >= 0:
        return (0, 200, 0)  # green (BGR)

    severity = min(abs(delta_inch) / 1.0, 1.0)
    green = int(200 * (1 - severity))
    red = int(255 * severity)
    return (0, green, red)


def draw_width_scan(mask, centers, perp, widths, edges):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    widths_inch = widths / PX_PER_INCH

    for i, (center, edge) in enumerate(zip(centers, edges)):
        delta_inch = 0.0 if i == 0 else float(widths_inch[i] - widths_inch[i - 1])
        color = delta_color(delta_inch)

        edge_lo, edge_hi = edge
        p1 = (center + perp * edge_lo).astype(int)
        p2 = (center + perp * edge_hi).astype(int)
        cv2.line(vis, tuple(p1), tuple(p2), color, 4)
        cv2.circle(vis, tuple(np.round(center).astype(int)), 4, (255, 255, 0), -1)

    return vis


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901207631.jpg"
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    mask, _ = extract_leaf(image)
    base, tip = find_base_tip(mask)
    centers, perp, widths, edges = measure_widths(mask, base, tip)
    vis = draw_width_scan(mask, centers, perp, widths, edges)

    out_path = "newmatch/test_output/width_scan.png"
    cv2.imwrite(out_path, vis)

    widths_inch = widths / PX_PER_INCH
    print(f"{'inch':>6} {'width_in':>10} {'delta_in':>10}")
    for i, w_in in enumerate(widths_inch):
        delta = 0.0 if i == 0 else w_in - widths_inch[i - 1]
        print(f"{i:6d} {w_in:10.2f} {delta:10.2f}")

    print(f"saved to {out_path}")
