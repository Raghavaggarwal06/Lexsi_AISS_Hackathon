# Phase 1 summary

**Verdict: GO**

Selected layer: hidden-state index 23 (hook on `model.model.layers[22]`), cleaned direction, chosen on the fit half by the largest margin over the two GO bars and a same-format bar (xstest_unsafe vs xstest_safe >= 0.9).

AUROC on the eval half at the selected layer:

- harmful vs benign: 0.964 (95% CI 0.940 to 0.984)
- harmful vs xstest_safe: 0.954 (95% CI 0.927 to 0.975)
- harmful vs orbench_hard: 0.909 (95% CI 0.872 to 0.941)
- xstest_unsafe vs xstest_safe (same format, not a go rule): 0.940 (95% CI 0.903 to 0.967)

False positive rate on the eval half at tau = -2.979 (90% TPR on fit harmful):

- benign: 0.056
- xstest_safe: 0.120
- orbench_hard: 0.296

True positive rate at that tau: harmful eval 0.877, xstest_unsafe eval 0.860.

For comparison, picking the layer on the eval half itself gives index 28 (cleaned): xstest 0.976, orbench 0.916, verdict GO. This is optimistic and is not the headline.

Selecting on the two GO bars alone picks index 11 (raw), which passes them (xstest 0.996, orbench 0.915) but gives only 0.844 on the same-format pair, so it was rejected.

Caution: the go-rule comparisons are partly solvable from surface text alone (see `confound.json`, surface baselines), so the same-format line is the one to trust. It was added to the layer choice after the first run showed the problem.
