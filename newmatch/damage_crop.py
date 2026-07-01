import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.width_scan import measure_widths, PX_PER_INCH


def find_damage_index(widths_inch, drop_thresh_inch=0.5):
    """Return the index of the first sample where width drops by more than
    drop_thresh_inch relative to the previous sample. If no such drop
    exists, returns len(widths_inch) (i.e. the whole leaf is undamaged)."""

    for i in range(1, len(widths_inch)):
        delta = widths_inch[i] - widths_inch[i - 1]
        if delta <= -drop_thresh_inch:
            return i

    return len(widths_inch)


def crop_undamaged(mask, base, tip, drop_thresh_inch=0.5, step_px=PX_PER_INCH):
    """Detect the first sharp width decrease along the leaf and crop the
    mask down to everything before that point (the undamaged region).
    Returns the cropped mask, the damage index (in inches from base), and
    the width profile used to make the decision."""

    centers, perp, widths, edges = measure_widths(mask, base, tip, step_px)
    widths_inch = widths / PX_PER_INCH

    damage_idx = find_damage_index(widths_inch, drop_thresh_inch)

    base = np.array(base, dtype=np.float64)
    tip = np.array(tip, dtype=np.float64)
    direction = tip - base
    length = np.linalg.norm(direction)
    direction /= length

    cutoff_dist = min(damage_idx * step_px, length)
    cutoff_point = base + direction * cutoff_dist

    # build a crop mask: keep only the region between base and the cutoff,
    # measured as projection along the base->tip direction
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    proj = (pts - base) @ direction

    keep = proj <= cutoff_dist
    cropped_mask = np.zeros_like(mask)
    cropped_mask[ys[keep], xs[keep]] = 255

    return cropped_mask, damage_idx, widths_inch, cutoff_point


def crop_to_length(mask, base, tip, length_inch, step_px=PX_PER_INCH):
    """Crop the mask to a fixed length (in inches) from base along the
    base->tip direction. Used to crop a healthy leaf to the same length as
    a query's undamaged crop for visual/shape comparison."""

    base = np.array(base, dtype=np.float64)
    tip = np.array(tip, dtype=np.float64)
    direction = tip - base
    direction /= np.linalg.norm(direction)

    cutoff_dist = length_inch * step_px

    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    proj = (pts - base) @ direction

    keep = proj <= cutoff_dist
    cropped_mask = np.zeros_like(mask)
    cropped_mask[ys[keep], xs[keep]] = 255

    return cropped_mask


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901207631.jpg"
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    mask, _ = extract_leaf(image)
    base, tip = find_base_tip(mask)
    cropped_mask, damage_idx, widths_inch, cutoff_point = crop_undamaged(mask, base, tip)

    print(f"damage detected at inch {damage_idx} (width={widths_inch[min(damage_idx, len(widths_inch)-1)]:.2f}in)")
    print(f"cutoff point: {cutoff_point}")

    out_path = "newmatch/test_output/damage_crop.png"
    cv2.imwrite(out_path, cropped_mask)
    print(f"saved to {out_path}")
