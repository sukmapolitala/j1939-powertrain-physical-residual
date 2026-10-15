# Interpretation of the repository's strict-causal results

These are source-specific measurements, **not** a statement of accuracy on new vehicles.

The primary analysis has 1,349 healthy fit samples, 450 calibration samples, 451 healthy test samples and 472 event samples from a single contiguous abnormal interval. An event sample is not an independent mechanical failure.

The proposed method achieved F1 = 0.876190, AUROC = 0.993832 and 0 false positives on the held-out healthy blocks; however the held-out healthy samples are temporal portions of the same original two recordings, not independent vehicles.

On the same strict-causal split, Isolation Forest had FAR = 0.348115, and One-Class SVM had FAR = 0.481153. The SVM's AUROC is higher than the proposed model's, a useful distinction between continuous anomaly ranking and fixed-threshold operational alarm decisions.

**Important negative results:** simultaneously injecting independent measurement noise at even 0.5 times the SPN resolution into all channels produces high false-positive rates. Complete loss of intake manifold pressure reduces recall sharply. Post-event recovery is outside the clean healthy operating envelope, and is not included as ordinary held-out normal data in the main study. All these are research limitations requiring further validation rather than hidden tuning opportunities.

Confidence intervals use one-second moving blocks to preserve short-range temporal dependence. Such resampling does **not** create additional independent engines or failure events.
