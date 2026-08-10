"""Execute the user-approved R1/R2 disk maintenance batches safely.

R1 losslessly gzip-compresses historical W&B binary logs and long JSONL
training streams. R2 replaces byte-identical standalone replay files with
hard links to their checkpoint-bundle copies, preserving both paths.

The script is intentionally resumable. A source is removed only after the
compressed stream has been read back and matched by byte count and SHA-256.
R2 changes a path only after both candidate files have matching SHA-256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/home/yjq/rl")
DEFAULT_AUDIT_DIR = Path(
    "/home/yjq/rl/CoCap1/cocap-voradj-speedopt/docs/audits"
)
PROTECTED_ROOTS = (
    Path(
        "/home/yjq/rl/CoCap1/cocap-voradj/artifacts/"
        "2026-08-08_200k_reference/stage4a"
    ),
    Path(
        "/home/yjq/rl/CoCap1/cocap-voradj/artifacts/"
        "2026-08-08_200k_reference/stage4c"
    ),
    Path(
        "/home/yjq/rl/CoCap1/cocap-voradj-speedopt/artifacts/"
        "2026-08-09_focal_replay_speed_validation"
    ),
    Path(
        "/home/yjq/rl/CoCap1/cocap-voradj-speedopt/artifacts/"
        "2026-08-09_200k_speedopt"
    ),
)
CHUNK_SIZE = 4 * 1024 * 1024


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_protected(path: Path) -> bool:
    absolute = path.absolute()
    return any(_is_relative_to(absolute, root) for root in PROTECTED_ROOTS)


def _sha256_and_size(stream: object) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(CHUNK_SIZE)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest, _ = _sha256_and_size(stream)
    return digest


def _append_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _r1_candidates(root: Path) -> list[Path]:
    selected: list[Path] = []
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(current_root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_protected(current / name)
        ]
        for filename in filenames:
            if not (
                filename.endswith(".wandb")
                or filename in {"training_metrics.jsonl", "episodes.jsonl"}
            ):
                continue
            path = current / filename
            if _is_protected(path) or path.is_symlink() or not path.is_file():
                continue
            selected.append(path)
    selected.sort(key=lambda item: item.stat().st_size, reverse=True)
    return selected


def _verify_existing_gzip(source: Path, destination: Path) -> tuple[str, int]:
    source_hash = _file_sha256(source)
    with gzip.open(destination, "rb") as stream:
        restored_hash, restored_size = _sha256_and_size(stream)
    if restored_hash != source_hash or restored_size != source.stat().st_size:
        raise RuntimeError(f"existing gzip does not match source: {destination}")
    return source_hash, restored_size


def _compress_one(source: Path, manifest: Path) -> tuple[int, int]:
    source_stat = source.stat()
    destination = source.with_name(source.name + ".gz")
    temporary = source.with_name(source.name + ".gz.tmp")

    if destination.exists():
        source_hash, restored_size = _verify_existing_gzip(source, destination)
        destination_size = destination.stat().st_size
        source.unlink()
        _append_manifest(
            manifest,
            {
                "action": "r1_recovered_existing_gzip",
                "source": str(source),
                "destination": str(destination),
                "original_bytes": restored_size,
                "compressed_bytes": destination_size,
                "sha256": source_hash,
            },
        )
        return restored_size, destination_size

    if temporary.exists():
        temporary.unlink()

    source_hash = hashlib.sha256()
    source_size = 0
    with source.open("rb") as input_stream, temporary.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=1,
            fileobj=raw_output,
            mtime=int(source_stat.st_mtime),
        ) as output_stream:
            while True:
                chunk = input_stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                source_hash.update(chunk)
                source_size += len(chunk)
                output_stream.write(chunk)
        raw_output.flush()
        os.fsync(raw_output.fileno())

    with gzip.open(temporary, "rb") as restored_stream:
        restored_hash, restored_size = _sha256_and_size(restored_stream)
    source_digest = source_hash.hexdigest()
    if source_size != source_stat.st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"source changed while compressing: {source}")
    if restored_size != source_size or restored_hash != source_digest:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"gzip verification failed: {source}")

    os.replace(temporary, destination)
    os.chmod(destination, source_stat.st_mode & 0o7777)
    os.utime(destination, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
    destination_size = destination.stat().st_size
    source.unlink()
    _append_manifest(
        manifest,
        {
            "action": "r1_gzip_verified",
            "source": str(source),
            "destination": str(destination),
            "original_bytes": source_size,
            "compressed_bytes": destination_size,
            "sha256": source_digest,
        },
    )
    return source_size, destination_size


def execute_r1(root: Path, audit_dir: Path, dry_run: bool) -> None:
    candidates = _r1_candidates(root)
    apparent = sum(path.stat().st_size for path in candidates)
    print(
        json.dumps(
            {
                "batch": "R1",
                "dry_run": dry_run,
                "candidate_count": len(candidates),
                "candidate_bytes": apparent,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if dry_run:
        return

    manifest = audit_dir / "R1_GZIP_MANIFEST_20260809.jsonl"
    original_total = 0
    compressed_total = 0
    skipped = 0
    for index, source in enumerate(candidates, start=1):
        try:
            original, compressed = _compress_one(source, manifest)
        except (OSError, RuntimeError) as error:
            skipped += 1
            _append_manifest(
                manifest,
                {
                    "action": "r1_skip_error",
                    "source": str(source),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                json.dumps(
                    {
                        "batch": "R1",
                        "index": index,
                        "total": len(candidates),
                        "source": str(source),
                        "status": "skipped",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            continue
        original_total += original
        compressed_total += compressed
        print(
            json.dumps(
                {
                    "batch": "R1",
                    "index": index,
                    "total": len(candidates),
                    "source": str(source),
                    "original_bytes": original,
                    "compressed_bytes": compressed,
                    "reclaimed_bytes": original - compressed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "batch": "R1",
                "status": "complete",
                "original_bytes": original_total,
                "compressed_bytes": compressed_total,
                "reclaimed_bytes": original_total - compressed_total,
                "skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _r2_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    cocap1 = root / "CoCap1"
    for standalone in cocap1.rglob("*_replay.pkl"):
        if (
            _is_protected(standalone)
            or standalone.is_symlink()
            or not standalone.is_file()
            or standalone.name == "replay.pkl"
        ):
            continue
        checkpoint_root = standalone.parent / "checkpoints"
        if not checkpoint_root.is_dir():
            continue
        standalone_size = standalone.stat().st_size
        matches = [
            candidate
            for candidate in checkpoint_root.glob("step_*/replay.pkl")
            if candidate.is_file()
            and not candidate.is_symlink()
            and candidate.stat().st_size == standalone_size
            and not _is_protected(candidate)
        ]
        if matches:
            pairs.append((standalone, sorted(matches)[-1]))
    pairs.sort(key=lambda item: item[0].stat().st_size, reverse=True)
    return pairs


def _deduplicate_one(
    standalone: Path,
    bundle: Path,
    manifest: Path,
) -> tuple[bool, int]:
    standalone_stat = standalone.stat()
    bundle_stat = bundle.stat()
    if standalone_stat.st_dev != bundle_stat.st_dev:
        return False, 0
    if standalone_stat.st_ino == bundle_stat.st_ino:
        return False, 0
    standalone_hash = _file_sha256(standalone)
    bundle_hash = _file_sha256(bundle)
    if standalone_hash != bundle_hash:
        _append_manifest(
            manifest,
            {
                "action": "r2_skip_hash_mismatch",
                "standalone": str(standalone),
                "bundle": str(bundle),
                "bytes": standalone_stat.st_size,
                "standalone_sha256": standalone_hash,
                "bundle_sha256": bundle_hash,
            },
        )
        return False, 0

    temporary = standalone.with_name(
        standalone.name + f".r2-hardlink-{os.getpid()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    os.link(bundle, temporary)
    os.replace(temporary, standalone)
    after = standalone.stat()
    if after.st_ino != bundle.stat().st_ino:
        raise RuntimeError(f"hard-link verification failed: {standalone}")
    _append_manifest(
        manifest,
        {
            "action": "r2_hardlink_verified",
            "standalone": str(standalone),
            "bundle": str(bundle),
            "bytes": standalone_stat.st_size,
            "sha256": standalone_hash,
            "old_standalone_inode": standalone_stat.st_ino,
            "bundle_inode": bundle_stat.st_ino,
            "new_link_count": after.st_nlink,
        },
    )
    return True, standalone_stat.st_size


def execute_r2(root: Path, audit_dir: Path, dry_run: bool) -> None:
    pairs = _r2_pairs(root)
    apparent = sum(standalone.stat().st_size for standalone, _ in pairs)
    print(
        json.dumps(
            {
                "batch": "R2",
                "dry_run": dry_run,
                "size_matched_pairs": len(pairs),
                "candidate_bytes": apparent,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if dry_run:
        return

    manifest = audit_dir / "R2_HARDLINK_MANIFEST_20260809.jsonl"
    changed = 0
    reclaimed = 0
    for index, (standalone, bundle) in enumerate(pairs, start=1):
        did_change, bytes_reclaimed = _deduplicate_one(
            standalone,
            bundle,
            manifest,
        )
        changed += int(did_change)
        reclaimed += bytes_reclaimed
        print(
            json.dumps(
                {
                    "batch": "R2",
                    "index": index,
                    "total": len(pairs),
                    "changed": did_change,
                    "standalone": str(standalone),
                    "bundle": str(bundle),
                    "reclaimed_bytes": bytes_reclaimed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "batch": "R2",
                "status": "complete",
                "changed_pairs": changed,
                "reclaimed_bytes": reclaimed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", choices=("r1", "r2", "all"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch in {"r1", "all"}:
        execute_r1(args.root, args.audit_dir, args.dry_run)
    if args.batch in {"r2", "all"}:
        execute_r2(args.root, args.audit_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
