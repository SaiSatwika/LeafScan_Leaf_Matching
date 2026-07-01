import cv2
import numpy as np

from leafmatchingmodel.core.extract_leaf import extract_leaf
from newmatch.base_tip import find_base_tip
from newmatch.damage_crop import crop_undamaged
from newmatch.build_profile_table import profiles_path, DEFAULT_ENV


def load_profile_table(path=None, env=DEFAULT_ENV):
    data = np.load(path or profiles_path(env), allow_pickle=False)
    areas = data["areas"] if "areas" in data else None
    return data["entry_ids"], data["profiles"], areas


def find_top_matches(query_profile, entry_ids, profiles, top_k=5):
    """Rank healthy leaves by L2 distance to the query's undamaged crop
    profile, using only the shared prefix length. Leaves shorter than the
    query profile can't be compared and are excluded (NaN -> inf).

    Returns a list of (entry_id, distance, index) sorted best-first, where
    index is the row index into profiles/areas (so callers can look up
    other per-leaf data, e.g. original_area, without a second search)."""

    n = len(query_profile)

    if profiles.shape[1] < n:
        return []

    candidate_prefix = profiles[:, :n]
    diff = candidate_prefix - query_profile[np.newaxis, :]
    has_nan = np.isnan(diff).any(axis=1)
    dists = np.linalg.norm(np.nan_to_num(diff), axis=1)
    dists[has_nan] = np.inf

    order = np.argsort(dists)
    top = order[:top_k]

    return [(str(entry_ids[i]), float(dists[i]), int(i)) for i in top if np.isfinite(dists[i])]


def expected_area_from_matching(query_profile, entry_ids, profiles, areas, top_k=5):
    """Find the top_k healthy leaves whose width profile most closely
    matches query_profile, and return the average of their recorded
    original_area (skipping any match with no recorded area).

    Returns (expected_area, matches) where matches is the list of
    (entry_id, distance, index) used, or (None, []) if no match has a
    usable area.
    """

    matches = find_top_matches(query_profile, entry_ids, profiles, top_k=top_k)
    if not matches or areas is None:
        return None, matches

    match_areas = [areas[i] for _, _, i in matches if not np.isnan(areas[i])]
    if not match_areas:
        return None, matches

    return float(np.mean(match_areas)), matches


def matching_confidence(damage_idx, expected_length):
    """How much of the leaf survived to the undamaged crop, relative to the
    expected full length of a healthy leaf of this leaf number (from
    bio_priors' mean_length). 1.0 = fully undamaged (crop covers the whole
    expected length), lower = more of the leaf was cropped away as damaged,
    so the area-by-matching estimate should be trusted less.

    expected_length should come from bio_priors[leaf_number]["mean_length"],
    NOT from the matched leaves' own lengths — width-profile similarity
    doesn't imply the leaves share a leaf number, and length varies by leaf
    number independently of width shape."""

    if not expected_length or expected_length <= 0:
        return 0.0
    return float(min(damage_idx / expected_length, 1.0))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "test/reconstruction_video_1772901207631.jpg"
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    mask, _ = extract_leaf(image)
    base, tip = find_base_tip(mask)
    _, damage_idx, widths_inch, _ = crop_undamaged(mask, base, tip)

    query_profile = widths_inch[:damage_idx]
    print(f"query undamaged profile ({damage_idx} inches): {np.round(query_profile, 2)}")

    entry_ids, profiles, areas = load_profile_table()
    print(f"loaded profile table: {len(entry_ids)} healthy leaves")

    expected_area, top_matches = expected_area_from_matching(query_profile, entry_ids, profiles, areas, top_k=5)

    print("\ntop 5 matches:")
    for entry_id, dist, i in top_matches:
        area = areas[i] if areas is not None else None
        print(f"  {entry_id}: distance={dist:.3f}  original_area={area}")

    print(f"\nexpected area (avg of top 5): {expected_area}")
