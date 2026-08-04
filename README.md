# CoCap Voronoi-Adjacency

Multi-agent reinforcement learning for cooperative capture and post-capture
coverage. The current release integrates CR-MS capture shaping, VCT-LS local
sensing/communication, and centroid-energy (CE) coverage in a reproducible
`4v1 scratch -> 8v2 -> 12v3` curriculum.

See the [Chinese release guide](docs/CRMS_VCTLS_CE_FINAL_RELEASE_20260804_ZH.md)
or [Chinese README](README.zh-CN.md) for the full method, artifact boundary,
and commands.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

python3 tools/supervise_crms_support_approach_curriculum.py \
  --train-device cuda:0 \
  --eval-device cuda:1
```

The release includes exactly three selected integrated checkpoints under
`artifacts/2026-08-04_crms_vctls_ce_final/`; rollout data and GIFs are excluded.
Faded trails are off by default and require an explicit `--draw-trails` flag.

ZoneDemo code, configurations, and tests are retained as an optional
generalization-evaluation branch. Zone is not enabled as a training scene in
the integrated main-line curriculum.
