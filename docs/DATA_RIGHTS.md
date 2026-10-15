# Permission and redistribution assessment

**Checked:** 15 October 2026.

**Upstream:** https://github.com/SystemsCyber/J1939_Attack_Datasets

The provider publishes its recordings in a public GitHub repository and documents the origin as testbench/research-truck recordings. A separate explicit dataset distribution licence or grant of permission to republish row-level transformed recordings **has not been confirmed** in the source repository. This does not establish that redistribution is prohibited in every jurisdiction; it means permission has not been positively demonstrated.

## Distribution decision

| Artifact | Public repository policy | Reason |
|---|---|---|
| Author-created analysis scripts and tests | Include | Our software output, MIT licence |
| Procedure / configuration / checksum / schematic diagram | Include after audit | Reproducibility without raw data |
| Aggregate precision, recall, F1, AUROC, AUPRC, event summaries | Include | New quantitative findings, avoiding substitution for source |
| Original `normal.log` and `labeled_TSC1.csv` | Do not include | Source-controlled data, licence unverified |
| Derived time-indexed physical SPN matrices (20/50/100 ms) | Hold pending authorization | Faithful transformations of the source recordings |
| Time-indexed model predictions / labels or fitted models | Hold pending review | May leak source behavior or annotation |
| Proprietary full SAE Digital Annex | Do not include | Licensed data |

**If permission is received**, retain attribution, exact source hashes, source licence text/permission record, version, and an explicit per-file data licence before releasing processed SPN matrices; release as a separately versioned data asset rather than silently changing the code package.

**Recommended request to source owners:** seek written permission specifically for redistribution of the decoded, resampled, time-aligned, physically interpreted SPN traces for `normal.log` and `labeled_TSC1.csv`, including derived normal/abnormal interval metadata, on GitHub and optionally Zenodo under a stated licence.
