from importlib import resources
from pathlib import Path
import tempfile
import shutil


def get_resource_path(package: str, resource: str) -> Path:
    """
    Returns a usable filesystem path for a packaged resource.
    Works even if installed as a zip/wheel.
    """

    with resources.as_file(resources.files(package) / resource) as p:
        return Path(p)


def copy_resource_to_tmp(package: str, resource: str) -> Path:
    """
    Some libs (like numpy/joblib) need real files.
    This ensures the file exists on disk.
    """

    src = resources.files(package) / resource

    tmp_dir = Path(tempfile.gettempdir()) / "leafmatchingmodel"
    tmp_dir.mkdir(exist_ok=True)

    dst = tmp_dir / resource

    if not dst.exists():
        with resources.as_file(src) as p:
            shutil.copy(p, dst)

    return dst