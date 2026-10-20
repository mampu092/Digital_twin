# Ethics, privacy, and release scope

This is a research proof of concept, not a certified medical device or deployment
for patient care. It does not diagnose disease, forecast imminent events, prescribe
medicines, recommend dosage changes or estimate causal treatment benefit.

Default patient records, SQLite demo state, screenshots and downloadable sample
reports are entirely invented. They do not show a real row from the source dataset.
Numeric source IDs do not prove de-identification.

Source licensing and legal permission could not be verified. The ZIP excludes the
CSV and row-level source derivatives. The code license grants rights only to authored
software and documentation; it grants no source-data rights. Before public release
confirm permission for the source data and trained weights. This package has not
been published, uploaded to GitHub, or shared with external services.

Any real-device extension requires consent, data minimization, authenticated
access, encryption in transit/at rest, access logs, retention policy, deletion
workflows, incident response, and institutional review appropriate to the use.
These deployment controls are not claimed as implemented in this local prototype.

Subgroup reports cover age and recorded gender where event counts permit. They
are descriptive and do not establish fairness, equal clinical utility, or performance
for underrepresented genders, locations, devices or clinical populations.

Unknown units, score directions and source linkage make wearable interpretation
especially uncertain. Anomaly detection evaluates engineered perturbations, not
clinical events. High synthetic accuracy is expected when the simulator produces
large coordinated shifts and is not evidence of clinical readiness.
