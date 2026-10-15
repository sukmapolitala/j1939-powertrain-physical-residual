# Protocol locked to the PPTR manuscript (strict-causal variant)

Primary test uses 20 ms grid. The independent normal run and pre-event segment are split forward in time in 60/20/20 proportions, then corresponding fit, calibration and healthy test subsets are pooled. The post-event segment is excluded from model fitting, calibration and primary test. The complete abnormal interval is unseen until final evaluation.

Expected primary sample counts for verified sources:

- fit: 1349
- healthy calibration: 450
- held-out healthy test: 451
- abnormal event test: 472
- event interval: 14.98–24.40 s relative to controlled-run timestamps

**Important:** this partition is not independent run-level validation. Healthy test blocks belong to the same two source runs; one annotated abnormal interval is one event, not 472 independent faults.

Conditional model: quadratic Huber model for six response SPNs given accelerator pedal, driver-demand torque and vehicle speed. Residual MAD, shrinkage covariance, dimension-normalized score, alpha=0.25 EWMA and three consecutive exceedances. Calibrate raw and EWMA thresholds separately on the held-out healthy calibration split, using q=0.99.

Baselines: Isolation Forest, One-Class SVM, same healthy fit/calibration periods and physical inputs, independent 99th percentile calibration.

Sensitivity experiments must not replace the preregistered primary 20 ms metric; report selection of any new candidate separately.
