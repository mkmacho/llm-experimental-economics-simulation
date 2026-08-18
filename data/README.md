# Data access

This public repository is code-only: it does not redistribute the experimental
microdata, participant-facing materials, or prepared tables.

To run `prepare`, download the official replication archive for Louis-Pierre
Lepage, *Experience-based Discrimination*, from
<https://www.openicpsr.org/openicpsr/project/192292/version/V1/view>. Confirm
the archive's current terms yourself, then place these files locally:

```text
data/raw/experience-based-discrimination/data.dta
data/raw/experience-based-discrimination/dataworker.dta
```

`prepare` recreates the normalized tables under `data/normalized/`. Both source
and normalized data are ignored by Git. Do not commit participant-level files or
provider interaction logs.
