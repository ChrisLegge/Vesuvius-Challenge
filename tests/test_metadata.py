from pathlib import Path

from vesuvius import validate_model_metadata


def test_model_metadata_files_are_valid():
    metadata_dir = Path("outputs/metadata")
    meta_files = sorted(metadata_dir.glob("model_*_meta.json"))

    assert len(meta_files) == 3
    for path in meta_files:
        metadata = validate_model_metadata(path)
        assert metadata["model_role"] in {"generalist", "anti_merge", "surface"}
