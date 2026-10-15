# Suggested upstream redistribution permission request

To the maintainers of `SystemsCyber/J1939_Attack_Datasets`:

We are conducting an academic study on normal-response modeling and physical-state anomaly detection in heavy-duty diesel powertrain telemetry from SAE J1939 signals. Our analysis uses `normal.log` and `labeled_TSC1.csv` from your publicly available repository and explicitly credits the original provider.

We would like to ask whether you authorize public redistribution in our separate GitHub/Zenodo research archive of **derived, time-indexed physical SPN tables**, produced by decoding and 20/50/100-ms resampling of the two source files. These tables would contain timestamps and physical variables (e.g., RPM, load, fuel rate, torque, intake pressure) with validity flags and a separate abnormal-state interval annotation; they would not contain raw CAN frames.

Could you confirm whether this is permitted, under which data licence, and whether any additional acknowledgements or citation text are required? We will not distribute the row-level transformed recordings until permission or another appropriate legal basis is confirmed.
