import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf


def find_base_tip(mask, n_rows=60):
    """Scan the mask in horizontal bands from top to bottom, recording the
    centroid x and width of each band. The band with greater width is the
    base (leaf attaches to the stem), the other end is the tip."""

    ys, _ = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()

    row_edges = np.linspace(y_min, y_max, n_rows + 1).astype(int)
    centers = []
    widths = []
    for i in range(n_rows):
        y0, y1 = row_edges[i], row_edges[i + 1]
        band = mask[y0:y1, :] > 0
        col_counts = band.sum(axis=0)
        band_xs = np.where(col_counts > 0)[0]
        if len(band_xs) == 0:
            continue
        cx = float(np.average(np.arange(mask.shape[1]), weights=col_counts))
        cy = (y0 + y1) / 2
        centers.append((cx, cy))
        widths.append(band_xs.max() - band_xs.min())

    centers = np.array(centers)
    widths = np.array(widths)

    # compare mean width of first vs last 15% of bands to decide which end is wider
    k = max(1, int(len(widths) * 0.15))
    top_width = widths[:k].mean()
    bottom_width = widths[-k:].mean()

    top_point = tuple(np.round(centers[0]).astype(int))
    bottom_point = tuple(np.round(centers[-1]).astype(int))

    if top_width >= bottom_width:
        base, tip = top_point, bottom_point
    else:
        base, tip = bottom_point, top_point

    return base, tip


def draw_base_tip(mask, base, tip):
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(mask_bgr, base, tip, (0, 0, 255), 6, tipLength=0.03)
    cv2.circle(mask_bgr, base, 10, (0, 255, 0), -1)
    cv2.circle(mask_bgr, tip, 10, (255, 0, 0), -1)
    return mask_bgr


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901909383.jpg"
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    mask, _ = extract_leaf(image)
    base, tip = find_base_tip(mask)
    vis = draw_base_tip(mask, base, tip)

    out_path = "newmatch/test_output/base_tip_mask.png"
    cv2.imwrite(out_path, vis)
    print(f"base: {base}, tip: {tip}")
    print(f"saved to {out_path}")
