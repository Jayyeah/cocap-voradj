# B1.5 Hidden-State / Local Target Evidence Reconstruction Audit

本报告是冻结 B1 teacher dataset 上的离线诊断；没有 online RL、没有修改正式 IQN observation/reward、没有实现 B3。

## 数据与重建充分性

- rows=146612，episodes=200，teacher SHA=2f39ea046acb56321585e7bbe1a8f0869a4dd85777ff2ae65ca29a7bd5680c89。
- direct sensing / friend adjacency / neighbor evidence 一致性：1.000000；friend token 匹配率：1.000000。
- teacher pursuing 与当前 direct visibility 是否完全重合：True；cross-tab（direct 0/1 × pursuing 0/1）=[[137139, 0], [0, 9473]]。
- history 按 episode → timestep → agent 顺序恢复，episode 起点 z=0；同步更新不使用同 step 递归。

## Role reconstruction（test）

| candidate | threshold | precision | recall | F1 | false persistence | premature release | timing error (steps) |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1_self_only | 0.81 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| H2_self_neighbor | 0.81 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.000 |
| H3_history_K3 | 0.05 | 0.9984 | 1.0000 | 0.9992 | 0.0001 | 0.0000 | 0.024 |
| H3_history_K5 | 0.22 | 0.9968 | 1.0000 | 0.9984 | 0.0002 | 0.0000 | 0.048 |
| H3_history_K10 | 1.00 | 0.9958 | 1.0000 | 0.9979 | 0.0003 | 0.0000 | 0.063 |

## Action probe（test）

| input | agreement | CE | KL(teacher‖probe) |
|---|---:|---:|---:|
| o | 0.5603 | 1.5654 | 1.3060 |
| o+m | 0.5621 | 1.5829 | 1.3269 |
| o+z | 0.5690 | 1.5450 | 1.3016 |
| o+history_K3 | 0.5739 | 1.7114 | 1.4919 |
| o+history_K5 | 0.5762 | 1.8087 | 1.6029 |
| o+history_K10 | 0.5761 | 2.0896 | 1.9048 |

H2-H1 overall action agreement delta=0.0069；support 条件 agreement delta=0.1033；support 条件 CE delta=-0.5528。
probe train split 未包含 action 4；该类在 9 类输出中以零概率补齐，未从 validation/test 借用标签。
phase/condition 明细与 train/validation 结果见 JSON。

## Near-observation aliasing（RMSE≤0.10）

| augmented vector | close pairs | mismatch rate | support mismatch | recovery mismatch |
|---|---:|---:|---:|---:|
| o | 118 | 0.7373 | 0.5000 | 1.0000 |
| o+m | 0 | 0.0000 | 0.0000 | 0.0000 |
| o+z | 0 | 0.0000 | 0.0000 | 0.0000 |
| o+history_K3 | 0 | 0.0000 | 0.0000 | 0.0000 |
| o+history_K5 | 0 | 0.0000 | 0.0000 | 0.0000 |
| o+history_K10 | 0 | 0.0000 | 0.0000 | 0.0000 |

说明：augmented RMSE 在扩展后的总维度上归一化；因此 close pairs 归零是固定阈值下的诊断信号，不能单独视为动作机制或因果证明，应结合 p50 距离与 action probe 解读。

## B1.5 结论

本数据集的 role reconstruction 是退化性结果：teacher pursuing 与当前 direct visibility 在全部 rows 上完全相等 （cross-tab=[[137139, 0], [0, 9473]]），因此 H1/H2 的完美或近乎完美 F1 不能检验 direct evidence 衰减、release memory 或隐藏 pursuing 状态。 H2 在 support 条件下仍使 test action agreement 提升 0.1033，并使 support 条件 CE 改变 -0.5528 （负值为改善）；但 overall agreement 提升不足 1 个百分点，故应保留 H2 作为候选而非宣称其已证明必要。 后续需要冻结 teacher 的轨迹中出现 direct=False、pursuing=True 的 rows，才能审计真正的 hidden-state reconstruction。 H2 does not materially improve teacher-role reconstruction (test F1 H1=1.0000, H2=1.0000).
