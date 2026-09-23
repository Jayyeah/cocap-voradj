# ROOT_CAUSE_PROOF — A1/A2 shared forward

Written before any production repair. See ROOT_CAUSE_PROOF.json for hashes and measured evidence.

- A1: step 760, third update, target_actor next_obs row 31, replay slot 1254, episode 1 agent 2.
- A2: step 808, fifteenth update, online actor next_obs row 69, replay slot 3099, episode 2 agent 3. The A2 finite count and finite max abs exactly match historical diag4.
- Both rows are inactive boundary-collision terminals, all 22 masks false, every observation value/type zero; weights and input finite.
- Same encoder file SHA256 in A1/A2: 567855d626403bcea549c4806ccae0951f57cc24540925d92ca6197d530b1de5.
- Exact forward: first fused Transformer layer output is non-finite. No module hooks or backend switches. Traced and untraced outputs agree exactly on finite entries and non-finite pattern; all four fused layers execute.
- Separate native-attention decomposition: norm1, QKV, Q/K/V and QK scores finite; all-masked softmax first produces NaN; native attention output and layer residual become NaN. Decomposed and fused layer outputs match exactly on finite entries and pattern on both CPU and CUDA.
- Single captured row fails deterministically at iteration 1 in 1/10/100/1000-run tests for BOTH actors. Full batch, reversed batch and duplicate row also fail. CPU also fails with identical weights/input/mask/mode/dtype.
- Five focused CPU regressions fail before repair.

## Why this is a production bug

Production runners deliberately encode missing inactive next_obs as all-zero arrays, including masks. The shared encoder forwards a fully blocked key set to native attention. Both learners compute next-state outputs before the terminated multiplier; 0 * NaN is NaN, so a terminal row contaminates loss/gradients. The expected terminal target is finite immediate reward. This is unrelated to entropy success/failure.

## Minimal repair contract

Clone the mask and retain slot 0 only for rows with no valid token, using that effective mask for attention/pooling. Do not modify stored observations, valid masks, weights, evader visibility, action mapping, termination/bootstrap rules or scientific hyperparameters. Every self-valid/otherwise nonempty observation takes bit-identical inputs and masks. Empty terminal placeholders have no physical next-state semantics; their arbitrary finite forward is excluded by the existing terminated rule. The repair prevents NaN at the source, with no nan_to_num, logits clamp, invalid-batch skipping or new backend flag.

The original A1 traceback has no global_step measurement: step-0 evaluation existence does not establish a step-0 failure. This audit measures the first target-forward failure at step 760.
