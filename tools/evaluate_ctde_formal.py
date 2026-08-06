"""Deterministic CTDE MASAC evaluation with configurable scene horizons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cocap_voradj.training.continuous.formal_config import (
    AW_ACTION_MODE,
    SCENES,
    resolve_formal_config,
    resolve_ladder_config,
)
from tools.run_continuous_ctde_training import (
    _make_trainer,
    _screen,
)


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs/experiments/continuous_marl_20260804/p6_formal_central_masac_4v1.yaml"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--seed", type=int, default=2026080601)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", default=str(FORMAL))
    args = parser.parse_args()
    raw = __import__("yaml").safe_load(Path(args.config).read_text(encoding="utf-8"))
    if str(raw.get("action", {}).get("mode", "")).strip().lower() in {AW_ACTION_MODE, "aw", "continuous_aw"}:
        config = resolve_ladder_config(args.config)
    else:
        config = resolve_formal_config(args.config)
    trainer = _make_trainer(config, args.device)
    payload = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.load_checkpoint(args.checkpoint, payload.get("contract", {}))
    scenes = tuple(item.strip() for item in args.scenes.split(",") if item.strip())
    unknown = set(scenes).difference(SCENES)
    if unknown:
        raise ValueError(f"unknown scenes: {sorted(unknown)}")
    screening = _screen(
        trainer,
        config,
        args.seed,
        episodes=args.episodes,
        device=args.device,
        scenes=scenes,
        max_steps=args.max_steps,
    )
    out = Path(args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(screening, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(screening, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
