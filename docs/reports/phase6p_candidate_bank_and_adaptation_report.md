# Phase 6P candidate bank and adaptation audit

The implementation generates the frozen 24-rule Phase 6C bank before exact destroyed-set deduplication. The P2 state contained 22 unique targets. Candidate identity and insertion order remained aligned across action tensorization and score-free feature construction; all targets received one score from each of the three fold-0 OOF seed models.

Ranking uses descending mean predicted continuation advantage and ascending `target_set_id` for ties. The first six ranks form the shortlist. For rank `r`, live target mass is `(1 / r) * mean(current destroy weights over all unique proposal origins)`. The selected target is sampled with the solver baseline RNG. The repair operator is then sampled independently from current repair weights. After the candidate outcome, reward 5, 1, or the 0.1 update floor is applied once to every unique destroy origin and once to the selected repair operator.

The probability audit increased `related` from 1.0 to 4.0. The affected rank-6 target probability changed from 0.068027211 to 0.080321285. Fixed-seed roulette selected `ts_bb341f6b39371e3257be` in both repetitions. Full ranks, scores, provenance, timings, adaptive weights, and iteration records are preserved in `outputs/phase6p_adaptive_portfolio_v1/smoke/result.json`.
