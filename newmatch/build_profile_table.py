import os
import time

import numpy as np

from newmatch.healthy_dataset import (
    get_healthy_reconstruction_paths,
    get_original_areas,
    DEFAULT_BASE_DIR,
    DEFAULT_ENV,
)
from newmatch.width_profile import compute_width_profile

TABLE_DIR = os.path.join(os.path.dirname(__file__), "files")


def profiles_path(env=DEFAULT_ENV, table_dir=TABLE_DIR):
    return os.path.join(table_dir, f"healthy_width_profiles_{env}.npz")


def build_and_save_table(db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV,
                         table_dir=TABLE_DIR, out_path=None):
    """Compute width profiles + original areas for every healthy leaf once,
    and save them as a padded 2D array + entry_id index + area vector so
    future lookups (shape matching AND area averaging) are just array ops
    (no image/artifact processing at query time).

    By default writes to newmatch's own files/ dir (for dev/standalone use
    in this repo). Pass out_path to write elsewhere, e.g. into the data
    node's <env_dir>/training_dataset/ alongside bio_priors.json."""

    paths = get_healthy_reconstruction_paths(db_path=db_path, base_dir=base_dir, env=env)

    entry_ids = []
    profiles = []

    t0 = time.time()
    for entry_id, path in paths.items():
        profile = compute_width_profile(path)
        if profile is not None and len(profile) > 0:
            entry_ids.append(entry_id)
            profiles.append(profile)
    elapsed = time.time() - t0

    if not profiles:
        raise RuntimeError("no healthy leaf width profiles could be computed")

    areas_by_id = get_original_areas(entry_ids, db_path=db_path, base_dir=base_dir, env=env)
    areas = np.array([areas_by_id.get(eid, np.nan) for eid in entry_ids], dtype=np.float64)
    n_with_area = int(np.sum(~np.isnan(areas)))

    max_len = max(len(p) for p in profiles)
    padded = np.full((len(profiles), max_len), np.nan, dtype=np.float64)
    for i, p in enumerate(profiles):
        padded[i, : len(p)] = p

    resolved_out_path = out_path or profiles_path(env, table_dir)
    os.makedirs(os.path.dirname(resolved_out_path), exist_ok=True)
    np.savez(resolved_out_path, profiles=padded, entry_ids=np.array(entry_ids), areas=areas)

    print(f"built profile table for {len(entry_ids)} healthy leaves in {elapsed:.1f}s "
          f"({n_with_area} with original_area)")
    print(f"saved to {resolved_out_path}")

    return resolved_out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=DEFAULT_ENV, choices=("prod", "dev"))
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--out-path", default=None,
                        help="Override output path (default: newmatch/files/healthy_width_profiles_<env>.npz)")
    args = parser.parse_args()

    build_and_save_table(base_dir=args.base_dir, env=args.env, out_path=args.out_path)
