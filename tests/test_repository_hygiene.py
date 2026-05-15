from pathlib import Path


FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".ckpt", ".tif", ".tiff", ".zip", ".npy", ".npz"}


def test_large_artifacts_are_not_tracked_in_working_tree():
    bad_paths = [
        path
        for path in Path(".").rglob("*")
        if ".git" not in path.parts and path.is_file() and path.suffix.lower() in FORBIDDEN_EXTENSIONS
    ]

    assert bad_paths == []
