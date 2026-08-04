# CR-MS + VCT-LS + CE final checkpoints

This directory contains only the three selected stage checkpoints for the final
integrated curriculum. It intentionally contains no rollout JSON, worker output,
or GIF.

| Stage | Scene | Selected checkpoint | SHA-256 |
| --- | --- | --- | --- |
| 1 | 4v1 scratch | `checkpoints/stage1_4v1_step_2000000.pt` | `2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89` |
| 2 | 8v2 course | `checkpoints/stage2_8v2_step_300000.pt` | `ef58ae9bdd018633afee0d16f2242d2f157a6cc18611e41474f115aed254b87e` |
| 3 | 12v3 course | `checkpoints/stage3_12v3_step_700000.pt` | `5e4173eac94c921685091d60033bffb8798180e974b72ec02437e3de265848ee` |

The current CE-first re-evaluation selected 300k as the best 8v2 checkpoint.
The supplied 12v3 checkpoint is the completed historical stage3 model, whose
actual training lineage used the then-selected 8v2 500k checkpoint. It has not
been represented as a retrain from 300k.

See `docs/CRMS_VCTLS_CE_FINAL_RELEASE_20260804_ZH.md` for reproduction and
evaluation commands.
