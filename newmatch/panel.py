import random

import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.damage_crop import crop_undamaged, crop_to_length
from newmatch.match_healthy import load_profile_table, find_top_matches
from newmatch.healthy_dataset import get_defoliated_reconstruction_paths
from newmatch.healthy_dataset import get_healthy_reconstruction_paths

CELL_W = 220
CELL_H = 700
PAD = 12
LABEL_H = 26


def load_mask(path):
    image = cv2.imread(path)
    if image is None:
        return None, None
    mask, _ = extract_leaf(image)
    return mask, image


def fit_to_cell(img_bgr, cell_w=CELL_W, cell_h=CELL_H - LABEL_H):
    h, w = img_bgr.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    y0 = (cell_h - new_h) // 2
    x0 = (cell_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_cell(img_bgr, label):
    thumb = fit_to_cell(img_bgr)
    label_bar = np.zeros((LABEL_H, CELL_W, 3), dtype=np.uint8)
    cv2.putText(
        label_bar, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack([label_bar, thumb])


def hstack_with_pad(cells, pad=PAD):
    h = cells[0].shape[0]
    spacer = np.zeros((h, pad, 3), dtype=np.uint8)
    out = cells[0]
    for c in cells[1:]:
        out = np.hstack([out, spacer, c])
    return out


def build_panel_row(entry_id, defoliated_path, entry_ids, profiles, healthy_paths, drop_thresh_inch=0.5):
    mask, image = load_mask(defoliated_path)
    if mask is None:
        return None

    base, tip = find_base_tip(mask)
    cropped_mask, damage_idx, widths_inch, _ = crop_undamaged(mask, base, tip, drop_thresh_inch)

    query_profile = widths_inch[:damage_idx]
    if len(query_profile) == 0:
        return None

    top_matches = find_top_matches(query_profile, entry_ids, profiles, top_k=5)

    cells = []
    cells.append(make_cell(image, f"target: {entry_id}"))
    cells.append(make_cell(cv2.cvtColor(cropped_mask, cv2.COLOR_GRAY2BGR), f"undamaged crop ({damage_idx}in)"))

    for match_id, dist, _ in top_matches:
        h_path = healthy_paths.get(match_id)
        if h_path is None:
            continue
        h_mask, h_image = load_mask(h_path)
        if h_mask is None:
            continue
        cells.append(make_cell(h_image, f"match: {match_id[-8:]} d={dist:.2f}"))

    for match_id, dist, _ in top_matches:
        h_path = healthy_paths.get(match_id)
        if h_path is None:
            continue
        h_mask, h_image = load_mask(h_path)
        if h_mask is None:
            continue
        h_base, h_tip = find_base_tip(h_mask)
        h_crop = crop_to_length(h_mask, h_base, h_tip, damage_idx)
        cells.append(make_cell(cv2.cvtColor(h_crop, cv2.COLOR_GRAY2BGR), f"crop: {match_id[-8:]}"))

    return hstack_with_pad(cells)


def main(n_samples=5, seed=0):
    random.seed(seed)

    defoliated_paths = get_defoliated_reconstruction_paths()
    healthy_paths = get_healthy_reconstruction_paths()
    entry_ids, profiles, _ = load_profile_table()

    sample_ids = random.sample(list(defoliated_paths.keys()), n_samples)

    rows = []
    for entry_id in sample_ids:
        print(f"processing {entry_id}...")
        row = build_panel_row(entry_id, defoliated_paths[entry_id], entry_ids, profiles, healthy_paths)
        if row is not None:
            rows.append(row)
        else:
            print(f"  skipped {entry_id} (no valid crop/mask)")

    if not rows:
        print("no valid rows produced")
        return

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)

    row_spacer = np.full((PAD, max_w, 3), 40, dtype=np.uint8)
    panel = padded_rows[0]
    for r in padded_rows[1:]:
        panel = np.vstack([panel, row_spacer, r])

    out_path = "newmatch/test_output/panel.png"
    cv2.imwrite(out_path, panel)
    print(f"saved panel to {out_path}")


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main(n_samples=n)
