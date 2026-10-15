# Data access, origin, and derived-data boundary

Source files must be obtained directly from the original provider:

- https://github.com/SystemsCyber/J1939_Attack_Datasets/blob/main/labeled_TSC1.csv
- https://github.com/SystemsCyber/J1939_Attack_Datasets/blob/main/normal.log

We use provider recordings as third-party research inputs. They include different operating records and controlled annotations; do not describe the entire source as multiple independently observed mechanical faults.

The physical-only datasets we calculate (`physical_normal_run_20ms.csv`, `physical_abnormal_run_20ms.csv`, etc.) **are derived, record-level representations of the source**. Removing message identifiers does not automatically confer a right to redistribute the time series. Until source rights are established, these files must remain local and are excluded by `.gitignore`.

The model consumes only physical variable measurements, and the label used to evaluate event scoring is supplied in a separate evaluation table.

See `SOURCE_SHA256.txt` for exact sources used by our analysis; hash mismatches indicate an upstream version change and must be investigated.
