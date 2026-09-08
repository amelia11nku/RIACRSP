# Phase 6O 预注册协议

状态：**`FROZEN_BEFORE_TARGETED_RELABEL_OR_OPTIMIZER_STEP`**。冻结路线：**`PROCEED_TOP_UTILITY_RETRAIN`**。

O0 证明 k≤6 的 neural shortlist exact/near-best recall 不足，因此 Route A 未启动。Primary 保留 Phase 6N candidate-conditioned pooler/fusion/heads，冻结完整 Phase 6F RT-HGT encoder；训练只使用 FP32、score-free inputs 和 whole-instance folds。

## Targeted relabeling

两条现有 CRN seeds 在 501/864 states 上给出不同 winner。冻结 union 为：original states 使用 Phase 6L top-4 ∪ Phase 6N top-4 ∪ fallback，new states 使用 Phase 6N top-4 ∪ fallback。实际 4,786 state-candidate rows，追加 seeds `[735101, 735102, 735103]`，共 14,358 additional continuation rows。所有 864 states 必须完成，不得按 outcome 停止；新标签单独保存，绝不覆盖 Phase 6L/6N truth。

Targeted candidates 训练时使用 5-seed mean，其他 candidates 保持 2-seed mean。Noise margin 仅从 training-fold CRN outcomes 拟合：candidate noise 为 1.4826×MAD，fold margin 为其 q75 并截断到 [0.0025, 0.03]。Held fold 不参与 margin、temperature、utility/regret scale、normalization 或 epoch selection。

## Primary objective

每个 source 获得 50% aggregate state weight，source 内再按 instance/state 平衡；opportunity weight 截断 [0.5, 2.0] 并在 source-instance 内归一化。Loss 为 top-set CE + near-best-vs-rest regret logistic + normalized soft expected utility，各权重 1.0；Huber 与 beats-fallback BCE 各 0.1。Phase 6N broad all-pairs loss和 standard-z ListNet 不用于 primary。

每个 outer fold/seed 使用对称 two-way inner validation，按 mean selected lift、near-best hit、top-1 regret、joint loss、earliest epoch 的冻结词典序选择一个 epoch，再在两个 outer-training folds 上 refit。

## Gate 与锁

Common raw lift 必须不低于 Phase 6L 的 0.0075163542，common regret 不高于 0.0338655744；expanded lift 与 grouped LCB 必须为正，S/M/L 不得反号，并保持 origin diversity、full bank、feasibility 和 fold isolation。失败为 `MODEL_REVISION_TOP_UTILITY`。

只有 OOF gate 通过后才能运行 3-seed、2N 的 non-promotable R12 development solver pilot。R13/R14 继续锁定；不运行 Gurobi。
