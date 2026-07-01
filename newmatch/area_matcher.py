import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.damage_crop import crop_undamaged
from newmatch.match_healthy import load_profile_table, expected_area_from_matching
from newmatch.build_profile_table import DEFAULT_ENV


class AreaMatcher:
    """Numbers-only leaf matching: no healthy images are fetched at
    inference time. Loads the cached healthy width-profile + original-area
    table once (a small sidecar file, e.g. bundled with the deployed model),
    then for each target leaf image:
      1. detects its undamaged region (base_tip + damage_crop)
      2. finds the top_k healthy leaves whose width profile most closely
         matches that undamaged region
      3. returns the average recorded original_area of those matches

    This is the "expected area given leaf matching" derived input for the
    leaf_matching CNN — a pure numeric lookup, no image I/O beyond the
    target leaf itself.
    """

    def __init__(self, profiles_path=None, env=DEFAULT_ENV):
        self.entry_ids, self.profiles, self.areas = load_profile_table(path=profiles_path, env=env)

    def expected_area(self, mask, top_k=5, drop_thresh_inch=0.5):
        """Given a leaf mask, return (expected_area, damage_idx, matches).
        expected_area is None if no top_k match has a recorded area."""

        base, tip = find_base_tip(mask)
        _, damage_idx, widths_inch, _ = crop_undamaged(mask, base, tip, drop_thresh_inch)

        query_profile = widths_inch[:damage_idx]
        if len(query_profile) == 0:
            return None, damage_idx, []

        expected_area, matches = expected_area_from_matching(
            query_profile, self.entry_ids, self.profiles, self.areas, top_k=top_k
        )
        return expected_area, damage_idx, matches

    def expected_area_from_image(self, image, top_k=5, drop_thresh_inch=0.5):
        mask, _ = extract_leaf(image)
        if mask is None:
            return None, 0, []
        return self.expected_area(mask, top_k=top_k, drop_thresh_inch=drop_thresh_inch)


if __name__ == "__main__":
    import sys
    import cv2

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901207631.jpg"
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    matcher = AreaMatcher()
    expected_area, damage_idx, matches = matcher.expected_area_from_image(image)

    print(f"damage_idx={damage_idx}in  expected_area={expected_area}")
    print("matches:")
    for entry_id, dist, i in matches:
        print(f"  {entry_id}  distance={dist:.3f}  area={matcher.areas[i]}")
