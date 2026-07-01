import json
import os
import sqlite3

DEFAULT_BASE_DIR = "/data/datasets/data_node/leafscan"
DEFAULT_ENV = "prod"


def _resolve_db_path(db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    if db_path:
        return db_path
    return os.path.join(base_dir, env, "artifact_index.sqlite")


def _query_reconstruction_paths(is_healthy, db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    resolved_db_path = _resolve_db_path(db_path, base_dir, env)

    con = sqlite3.connect(resolved_db_path)
    cur = con.cursor()
    cur.execute(
        """
        SELECT entry_id, artifact_dir
        FROM artifacts
        WHERE level='leaf' AND artifact='reconstruction' AND is_healthy=?
        """,
        (is_healthy,),
    )
    rows = cur.fetchall()
    con.close()

    paths = {}
    for entry_id, artifact_dir in rows:
        img_path = os.path.join(artifact_dir, "latest.jpg")
        if os.path.exists(img_path):
            paths[entry_id] = img_path

    return paths


def get_healthy_reconstruction_paths(db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    """Query the artifact index for healthy leaves and return a dict of
    entry_id -> path to the latest reconstruction image."""
    return _query_reconstruction_paths(1, db_path, base_dir, env)


def get_defoliated_reconstruction_paths(db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    """Query the artifact index for non-healthy (defoliated) leaves and
    return a dict of entry_id -> path to the latest reconstruction image."""
    return _query_reconstruction_paths(0, db_path, base_dir, env)


def get_original_areas(entry_ids, db_path=None, base_dir=DEFAULT_BASE_DIR, env=DEFAULT_ENV):
    """Look up the recorded 'original_area' artifact (results.original_area)
    for each entry_id. Returns a dict of entry_id -> float, omitting entries
    with no original_area artifact or unreadable JSON."""

    resolved_db_path = _resolve_db_path(db_path, base_dir, env)

    con = sqlite3.connect(resolved_db_path)
    cur = con.cursor()
    placeholders = ",".join("?" * len(entry_ids))
    cur.execute(
        f"""
        SELECT entry_id, artifact_dir
        FROM artifacts
        WHERE level='leaf' AND artifact='original_area' AND entry_id IN ({placeholders})
        """,
        list(entry_ids),
    )
    rows = cur.fetchall()
    con.close()

    areas = {}
    for entry_id, artifact_dir in rows:
        json_path = os.path.join(artifact_dir, "latest.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
            area = data.get("results", {}).get("original_area")
            if area is not None:
                areas[entry_id] = float(area)
        except (json.JSONDecodeError, OSError):
            continue

    return areas


if __name__ == "__main__":
    paths = get_healthy_reconstruction_paths()
    print(f"found {len(paths)} healthy leaf reconstructions")
    for entry_id, path in list(paths.items())[:5]:
        print(entry_id, path)

    defoliated = get_defoliated_reconstruction_paths()
    print(f"found {len(defoliated)} defoliated leaf reconstructions")
