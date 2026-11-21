"""Model weights: downloaded from the Hugging Face Hub on first use and cached there.

Each stage has its own model repo:

    transcriber  kobimusic/dettato
    evidence     kobimusic/dettato-evidence
    grouper      kobimusic/dettato-grouper
    drums        kobimusic/dettato-drums

Set ``DETTATO_WEIGHTS=/some/dir`` to load from local copies instead (``/some/dir/dettato/config.json``,
``/some/dir/dettato-grouper/model.safetensors``, ...), for example on a machine without network access.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPOS = {
    "transcriber": "kobimusic/dettato",
    "evidence": "kobimusic/dettato-evidence",
    "grouper": "kobimusic/dettato-grouper",
    "drums": "kobimusic/dettato-drums",
}


def fetch(stage: str, filename: str, revision: str | None = None) -> str:
    """Local path of ``filename`` in the stage's model repo (downloaded once, then read from the cache)."""
    repo = REPOS[stage]
    local = os.environ.get("DETTATO_WEIGHTS")
    if local:
        path = Path(local).expanduser() / repo.split("/", 1)[1] / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} (DETTATO_WEIGHTS is set, so the Hub is not used)")
        return str(path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, revision=revision)


def load_config(stage: str, filename: str = "config.json") -> dict:
    with open(fetch(stage, filename)) as fh:
        return json.load(fh)


def load_state(stage: str, filename: str = "model.safetensors") -> dict:
    from safetensors.torch import load_file

    return load_file(fetch(stage, filename))
