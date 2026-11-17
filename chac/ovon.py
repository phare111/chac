from pathlib import Path


OVON_SPLIT_FILES = {
    "val_seen": "val_seen.json.gz",
    "val_seen_synonyms": "val_unseen_easy.json.gz",
    "val_unseen": "val_unseen_hard.json.gz",
}


def ovon_split_data_path(data_root: Path, split: str) -> Path:
    try:
        filename = OVON_SPLIT_FILES[split]
    except KeyError as exc:
        raise ValueError(
            f"OVON split must be one of {sorted(OVON_SPLIT_FILES)}"
        ) from exc
    return data_root / split / filename
