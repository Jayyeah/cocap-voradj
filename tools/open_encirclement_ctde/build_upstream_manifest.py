#!/usr/bin/env python3
"""Build the immutable upstream source/checkpoint manifest."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from open_encirclement_ctde.runtime import atomic_json_dump, sha256_file


REPOSITORIES = {
    "light_mappo": {
        "url": "https://github.com/tinyzqh/light_mappo.git",
        "branch": "main",
        "commit": "c503d89b6f28c9687ce9e45304fe66b57322ce1e",
        "license": "MIT",
        "entrypoints": ["train/train.py", "scripts/render_uav.py"],
        "dependency_files": ["requirements.txt"],
        "role": "MAPPO engineering base and UAV adapter source",
    },
    "MADDPG_Multi_UAV_Roundup": {
        "url": "https://github.com/reinshift/MADDPG_Multi_UAV_Roundup.git",
        "branch": "main",
        "commit": "15309de231f639c62d2049b0ad5b07b8975309c9",
        "license": "MIT",
        "entrypoints": ["main.py", "main_evaluate.py"],
        "dependency_files": ["README.md"],
        "role": "Upstream-Raw environment, MADDPG, and supplied checkpoints",
    },
    "KF_AA_MARL": {
        "url": "https://github.com/reinshift/KF_AA_MARL.git",
        "branch": "main",
        "commit": "c8d68cab016ce9e6b18f8435b2faa0048d59da41",
        "license": "MIT",
        "entrypoints": ["run.py", "src/main.py", "src/test_model.py"],
        "dependency_files": ["requirements.txt"],
        "role": "Deferred multi-target/MATD3 donor; not used by L0",
    },
}


def build(root: Path) -> dict:
    repos = {}
    for name, metadata in REPOSITORIES.items():
        snapshot = root / name / metadata["commit"]
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
        files = []
        for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
            files.append(
                {
                    "path": str(path.relative_to(snapshot)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        license_path = snapshot / "LICENSE"
        models = [
            item for item in files
            if item["path"].startswith(("tmp/", "model/"))
            and not item["path"].endswith((".csv", ".png"))
        ]
        repos[name] = {
            **metadata,
            "snapshot_path": str(snapshot),
            "snapshot_file_count": len(files),
            "snapshot_bytes": sum(item["size"] for item in files),
            "license_sha256": sha256_file(license_path),
            "git_lfs_files": [],
            "supplied_models": models,
            "files": files,
        }
    return {
        "schema": "open-encirclement-upstream-manifest-v1",
        "fetch_date": "2026-08-23",
        "generated_at": datetime.now().astimezone().isoformat(),
        "claim_boundary": (
            "These repositories are software donors, not official code for a corresponding "
            "encirclement paper. Upstream-Raw, Audit-Corrected, and CoCap-Bridge evidence must not mix."
        ),
        "repositories": repos,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.root.resolve())
    atomic_json_dump(manifest, args.output.resolve())
    print(json.dumps({"output": str(args.output), "repositories": len(manifest["repositories"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
