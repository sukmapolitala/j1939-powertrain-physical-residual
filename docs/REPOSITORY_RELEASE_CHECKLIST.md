# Public release checklist

- [x] Decoder and strict-causal experiment script recover the verified source sample counts (21,568 / 19,957).
- [x] No post-event segment is included in model fit/calibration for the primary experiment.
- [x] Main 20-ms benchmark reproduces F1 = 0.876190 and 0/451 healthy false alarms.
- [x] Isolation Forest and One-Class SVM use the same healthy fit/calibration partitions.
- [x] Original upstream file hashes are recorded.
- [x] Git ignores original recordings and row-level decoded telemetry.
- [x] Regression tests for leakage and chronological splitting pass.
- [ ] Verify installation and source reproduction from another computer/clean environment.
- [ ] Review author ownership, affiliations, and final software copyright notice.
- [ ] Obtain permission before adding any derived record-level SPN CSV files.
- [ ] Fill authors and manuscript DOI in a machine-readable citation file after author approval.
- [ ] Create GitHub repository and confirm code has been pushed/visible.
- [ ] Freeze release/tag and consider Zenodo archiving after contents review.
