# NGAS A1.7A-R prior-staleness rejected protocol v1

The CUDA implementation smoke failed before formal execution because its three-action
slice did not contain the archived production-selected action, while the selected-action
quality assertion required that action. No formal raw record or formal lock was created.

The v1 protocol and failure metadata are preserved. Revision 2 may change only the
explicitly partial smoke path; the formal path must still enumerate the full persistent
bank and fail closed when an archived selected action is absent. R13/R14 remained locked,
CORE45 remained excluded, and no Gurobi run occurred.
