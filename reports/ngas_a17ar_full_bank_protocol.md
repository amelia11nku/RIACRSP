# NGAS A1.7A-R full-bank utility protocol

This protocol was frozen before full-bank U0-U3 outcomes. It selects exactly
**18 clean non-R12 states** and **18 R12 development-exposed audit-only states**.
Every state evaluates the true frozen production joint-action bank with eight
matched stochastic repair/decode trials per action.

U1 forces the audited best-of-eight action, reconstructs the archived online
portfolio state, and continues exactly two production iterations. Because the
horizon is shorter than the fixed refresh interval of 20, the captured critic
prior and action bank remain persistent and no refresh is introduced.

- Checkpoint: `448b0aaf871f0629dec2d94bad63c888fbdaf71c113eb5ade58c4228647c8560`
- Candidate bank: `e64c7a1cdfbe64495a2e3175fe7060dcea745f094ea6e3f1b3839fb48367bd07`
- Freeze source commit: `c477026f96cc683fd06ff4b77db34a4e4bfbc879`
- R13/R14: locked; CORE45: no access; Gurobi: false.
