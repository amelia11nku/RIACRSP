# NGAS A1.7A-S utility-support transitions

`nonconstant` and `positive-best` are separate properties. Every transition rate
uses states satisfying its named source condition as the denominator.

| transition | numerator | denominator | fraction |
|---|---|---|---|
| U0_nonconstant_to_U1_nonconstant | 16 | 16 | 1.0000 |
| U0_constant_to_U1_nonconstant | 2 | 11 | 0.1818 |
| U0_constant_to_U1_constant | 9 | 11 | 0.8182 |
| U0_nonconstant_to_U3_nonconstant | 16 | 16 | 1.0000 |
| U0_constant_to_U3_nonconstant | 11 | 11 | 1.0000 |
| U0_positive_best_to_U1_positive_best | 16 | 16 | 1.0000 |
| not_U0_positive_best_to_U1_positive_best | 2 | 11 | 0.1818 |

## Explicit transition state IDs

```json
{
  "U0_constant_to_U1_nonconstant": [
    "CLEAN|CB1_TRAIN_L_CF1_RI3_TI3_R01:seed817000008:iteration1640",
    "CLEAN|CB1_TRAIN_S_CF1_RI1_TI1_R04:seed817000099:iteration580"
  ],
  "U0_constant_to_U3_nonconstant": [
    "CLEAN|CB1_TRAIN_L_CF1_RI3_TI3_R01:seed817000008:iteration1640",
    "CLEAN|CB1_TRAIN_S_CF1_RI1_TI1_R04:seed817000099:iteration580",
    "CLEAN|CB1_TRAIN_S_CF1_RI2_TI3_R01:seed817000059:iteration4720",
    "CLEAN|CB1_TRAIN_M_CF1_RI2_TI3_R02:seed817000032:iteration3560",
    "CLEAN|CB1_TRAIN_S_CF1_RI1_TI1_R01:seed817000054:iteration2760",
    "CLEAN|CB1_TRAIN_S_CF1_RI2_TI3_R01:seed817000059:iteration2640",
    "CLEAN|CB1_TRAIN_M_CF1_RI2_TI3_R04:seed817000091:iteration3300",
    "CLEAN|CB1_TRAIN_M_CF1_RI1_TI2_R03:seed817000028:iteration2160",
    "CLEAN|CB1_TRAIN_S_CF1_RI3_TI3_R04:seed817000101:iteration4940",
    "CLEAN|CB1_TRAIN_L_CF1_RI1_TI1_R03:seed817000000:iteration2980",
    "CLEAN|CB1_TRAIN_L_CF2_RI1_TI1_R04:seed817000084:iteration1680"
  ],
  "not_U0_positive_best_to_U1_positive_best": [
    "CLEAN|CB1_TRAIN_L_CF1_RI3_TI3_R01:seed817000008:iteration1640",
    "CLEAN|CB1_TRAIN_S_CF1_RI1_TI1_R04:seed817000099:iteration580"
  ]
}
```

Scale and stage rows are stored in the machine-readable CSV.
