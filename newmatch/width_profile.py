import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.width_scan import measure_widths, PX_PER_INCH


def compute_width_profile(image_path, step_px=PX_PER_INCH):
    """Load an image, extract the leaf mask, find base/tip, and return the
    full width-per-inch profile (in inches) from base to tip."""

    image = cv2.imread(image_path)
    if image is None:
        return None

    mask, _ = extract_leaf(image)
    if mask is None:
        return None

    base, tip = find_base_tip(mask)
    _, _, widths, _ = measure_widths(mask, base, tip, step_px)
    return widths / PX_PER_INCH
