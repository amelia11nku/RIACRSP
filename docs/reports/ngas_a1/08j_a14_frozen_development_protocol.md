# NGAS-A1.4 小规模 R12 开发协议冻结

协议已在查看任何正式运行结果前冻结。

- 实现提交：`7bb5983cd77cca446c4e2e3aa3b896e125b8e0db`
- 协议：`outputs/ngas_a1/search_integration_v1/preregistration/protocol.json`
- 协议 SHA256：`7287f8c497dfe649818b648748e5df032275addfe05156f030c1f242e1b71ee4`
- 配置 SHA256：`3aec8be1bbee7bce32e1e2f93a601153d8de9add90409c59334c0b24bf13a182`
- CUDA 烟测 SHA256：`aa66fd0c8d0b5cc3cf7d0d6a5509cd826a686cd532556485e9b2e06802730048`
- C1 checkpoint SHA256：`448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`

协议锁定 54 个源码与入口文件的内容哈希、三个 R12 实例及其内容哈希、三个开发种子、六组消融、运行顺序、墙钟预算、搜索超参数、生产候选范围和选择规则。预计执行 54 次主试验；每个实例—种子块按六种模式相邻执行。

正式主试验只决定固定刷新和事件刷新中是否存在合格生产机制。任何机制选择必须同时通过可行回放、持续 critic 影响、每次 critic 调用的有效引导跨度和对在线组合器内部对照的预注册质量规则。若两个候选均不通过，结果直接记为 `NGAS_A1_REVISE_SEARCH_INTEGRATION`。

机制若成功冻结，下一步仍是预先写入协议的 seed-matched、held-fold-0 C1/R1 解释性消融。R1 始终无生产资格。R13、R14、Gurobi 和 git push 保持禁用。
