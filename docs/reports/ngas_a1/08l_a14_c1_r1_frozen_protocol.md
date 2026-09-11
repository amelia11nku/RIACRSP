# NGAS-A1.4 C1/R1 解释性消融协议冻结

固定刷新机制已由 A1.4 主试验冻结。表示消融协议在任何 C1/R1 正式求解结果产生前固定如下：

- 实现提交：`aeb3c2b230bbfc383da1dd382c3b2109a0239578`
- 协议：`outputs/ngas_a1/search_integration_c1_r1_v1/preregistration/protocol.json`
- 协议 SHA256：`1bbe5f9c6b26e4d54090eb1d9fd4d5af4f493a2ed4376311f25b94deb11c0325`
- 继承的 A1.4 主协议 SHA256：`7287f8c497dfe649818b648748e5df032275addfe05156f030c1f242e1b71ee4`
- CUDA 烟测 SHA256：`22a97850252927a312f7995a1607a915c229cd9e428d04dd672691235b9e48a3`

试验仅改变表示 checkpoint。C1 与 R1 使用相同 `PERSISTENT_FIXED_REFRESH` 机制、三个实例、三个种子、held fold 0、搜索参数、预算、运行路径和遥测，共 18 次运行。每个种子的 C1/R1 checkpoint 路径与内容哈希均已写入协议。

该比较只解释表示层在完整求解器中的影响。C1 保持生产身份，R1 无论最终质量如何都不能晋升，固定刷新机制也不会因该比较被重新选择。只有 18/18 完成、哈希闭合、终态精确回放与完成审计通过后，A1.5 才可解锁。

R13、R14 和 Gurobi 保持锁定。
