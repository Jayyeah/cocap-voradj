# A1 entropy-only Local-Q Coverage independent evaluation

- Evaluation branch: `evaluation/a1-entropy-localq-conclusion-20260923`
- Training/inference base: `78adf7889900affa7dd62aa261b36a7c8b55bb15`
- Astra shared A1/A2 terminal-row fix: `4e210c090ec696bda73ff36877f8e33270af86a`
- Scene: Pure Coverage / `voradj_coverage`
- Reset seeds: `2026097101..2026097120`, identical for every checkpoint and mode
- Native horizon: 3000 decision steps = 1500 seconds
- Strict CE: current centroid-energy strict RMS/max + area/center/inside thresholds with 30-step hold
- Collision: `synchronized_swept_v1`
- Parameter updates: 0

Each checkpoint has 20 argmax and 20 sampled complete rollouts. The 0-step actor was reconstructed from the fresh training seed because the formal runner did not save a step-0 model file. The evaluation used the corrected observation/CUDA terminal-row runtime and did not update parameters.

`CE`, `collision`, `failure`, and `censored` are counts out of 20. Numeric triples are mean / median / p90. `time` uses successful episodes only. `hold` is strict hold steps over all episodes.

## 40-rollout summaries

| checkpoint | mode | CE | collision | failure / censored | CE RMS | CE max | area CV | hold | time s | return | policy entropy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | argmax | 0/20 | 4/20 | 4/16 | .2927/.2941/.3685 | .3866/.3836/.4940 | .4725/.4148/.8215 | 0/0/0 | -/-/- | -14970.8/-9757.5/-3720.2 | 2.0862/2.0849/2.0907 |
| 0 | sample | 0/20 | 20/20 | 20/0 | .2886/.2807/.4009 | .3464/.3204/.5046 | .3322/.3098/.6897 | 0/0/0 | -/-/- | -2867.4/-1768.3/-734.9 | 2.0902/2.0896/2.0947 |
| 25k | argmax | 0/20 | 6/20 | 6/14 | .2707/.2783/.3665 | .3651/.3701/.5146 | .4147/.4300/.7120 | 0/0/0 | -/-/- | -10621.8/-10763.7/-479.0 | .7751/.8067/1.2723 |
| 25k | sample | 0/20 | 20/20 | 20/0 | .2386/.2217/.3598 | .2800/.2916/.3730 | .2396/.1589/.5245 | 0/0/0 | -/-/- | -742.2/-690.4/-404.3 | 1.0360/1.0478/1.3071 |
| 50k | argmax | 0/20 | 6/20 | 6/14 | .1245/.0998/.2043 | .1636/.1213/.2649 | .2597/.2350/.4622 | 0/0/0 | -/-/- | -912.9/-898.6/-352.7 | 1.8514/2.1640/2.1740 |
| 50k | sample | 12/20 | 5/20 | 5/3 | .0825/.0468/.1306 | .1123/.0733/.1637 | .1290/.0762/.2408 | 18/30/30 | 949.6/1054.5/1351.2 | -1171.2/-692.3/-355.6 | 1.9795/2.1372/2.1494 |
| 75k | argmax | 9/20 | 7/20 | 7/4 | .1092/.0596/.2881 | .1540/.0892/.3772 | .2078/.0800/.4504 | 13.5/0/30 | 59.8/63.5/79.5 | -1741.9/-347.8/-43.3 | 1.6780/1.7831/2.1692 |
| 75k | sample | 12/20 | 8/20 | 8/0 | .1261/.0460/.3121 | .1716/.0719/.3918 | .2503/.1571/.4434 | 18/30/30 | 102.1/103.8/137.7 | -349.4/-280.7/-66.7 | 1.6178/1.8083/1.9954 |
| 100k | argmax | 13/20 | 5/20 | 5/2 | .0665/.0413/.1081 | .0877/.0525/.1408 | .1114/.0708/.2015 | 19.5/30/30 | 60.1/56.0/77.7 | -218.4/-194.0/-42.7 | 1.7941/1.8530/1.9777 |
| 100k | sample | 20/20 | 0/20 | 0/0 | .0401/.0417/.0479 | .0549/.0540/.0697 | .0728/.0601/.1024 | 30/30/30 | 108.5/92.0/166.6 | -257.8/-154.7/-41.8 | 1.9491/1.9251/2.0728 |

## AW9 action histogram counts, actions 0..8

| checkpoint | argmax | sample |
|---|---|---|
| 0 | `[195823,0,0,0,0,0,1,0,1116]` | `[5653,1834,3286,2433,1291,2667,3791,1667,4438]` |
| 25k | `[27,192,2575,1899,2143,20694,165,144,141709]` | `[631,934,1031,995,1105,1539,657,1056,6216]` |
| 50k | `[72897,0,18109,4472,6356,59467,630,927,6562]` | `[19246,15548,19701,15013,16053,18837,11209,10051,13138]` |
| 75k | `[436,0,43,447,49443,247,649,668,1563]` | `[1101,950,1017,1355,1152,1263,1158,1408,2180]` |
| 100k | `[7749,0,8146,38,7507,343,1840,1347,4210]` | `[1718,1652,1781,1522,1675,1895,2477,2272,2360]` |

## Trend and conclusion

`A1_SUSTAINED_PARTIAL_LEARNING`

This is not a single accidental 75k success. The signal is already visible at 50k in sampled mode (`12/20`), is repeated at 75k (`9/20` argmax and `12/20` sample), and improves at 100k (`13/20` argmax and `20/20` sample). CE RMS/max and area CV also improve materially by 100k. Therefore:

- 75k is a real learning signal, not chance.
- 100k is not a regression; it is the strongest checkpoint in this evaluation.
- The result is sustained partial learning, not deterministic 100% argmax success: 100k argmax still has `13/20` CE success, `5/20` collision, and `2/20` censored episodes.

The detailed raw per-episode JSON and evaluator source remain in the private evaluation worktree; this public branch contains the conclusion report only.
