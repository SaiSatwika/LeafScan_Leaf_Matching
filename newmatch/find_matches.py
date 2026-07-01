import cv2

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.damage_crop import crop_undamaged
from newmatch.match_healthy import load_profile_table, find_top_matches
from newmatch.healthy_dataset import get_healthy_reconstruction_paths, DEFAULT_BASE_DIR, DEFAULT_ENV


class HealthyMatcher:
    """Loads the healthy-leaf profile table and path index once, then serves
    many match queries cheaply. This is a dev/visualization helper (used by
    panel.py to show the actual matched healthy leaf images) — it still
    resolves reconstruction image paths, which requires local disk access
    to the healthy dataset. For inference-time area estimation (no images
    needed), use AreaMatcher instead."""

    def __init__(self, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
        self.entry_ids, self.profiles, self.areas = load_profile_table(env=env)
        self.healthy_paths = get_healthy_reconstruction_paths(base_dir=base_dir, env=env)

    def match(self, image_path, top_k=5, drop_thresh_inch=0.5):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        mask, _ = extract_leaf(image)
        if mask is None:
            raise ValueError(f"Leaf extraction failed for: {image_path}")

        base, tip = find_base_tip(mask)
        _, damage_idx, widths_inch, _ = crop_undamaged(mask, base, tip, drop_thresh_inch)

        query_profile = widths_inch[:damage_idx]
        if len(query_profile) == 0:
            raise ValueError(f"No undamaged region detected for: {image_path}")

        top_matches = find_top_matches(query_profile, self.entry_ids, self.profiles, top_k=top_k)

        return [
            {"entry_id": entry_id, "path": self.healthy_paths.get(entry_id), "distance": dist}
            for entry_id, dist, _ in top_matches
        ]


def find_matching_healthy_leaves(image_path, top_k=5, drop_thresh_inch=0.5, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    """Given a path to a leaf reconstruction image, detect its undamaged
    (pre-damage) region and return the top_k healthy leaves whose same-length
    crop most closely matches its width profile.

    Returns a list of dicts, sorted best-match first:
        {"entry_id": str, "path": str, "distance": float}

    For matching many leaves in a loop, use HealthyMatcher instead — this
    function reloads the profile table and healthy path index on every call.
    """
    return HealthyMatcher(base_dir=base_dir, env=env).match(
        image_path, top_k=top_k, drop_thresh_inch=drop_thresh_inch
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901207631.jpg"
    matches = find_matching_healthy_leaves(path)

    print(f"top {len(matches)} healthy matches for {path}:")
    for m in matches:
        print(f"  {m['entry_id']}  distance={m['distance']:.3f}  path={m['path']}")
