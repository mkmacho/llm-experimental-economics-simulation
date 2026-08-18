# Public-release and data-governance note

## Status

The pipeline code and its technical documentation are appropriate to showcase.
This public-release version is code-only and must not include third-party
experimental materials or subject-level microdata.

## Data and material provenance

The `Experience-based Discrimination` inputs originate from Louis-Pierre
Lepage's public replication archive on
[openICPSR](https://www.openicpsr.org/openicpsr/project/192292/version/V1/view).
The source archive—not this repository—is the authoritative location for the
paper's data, experimental materials, and reuse terms. The paper is cited in the
[README](README.md#references).

The raw files include a stable study participant identifier (`id`) and demographic
fields such as age, gender, race, education, state, employment, household status,
income, and marital status. They are not distributed here.

## Checklist before a public release

1. Keep the upstream citation and link in the README and this note.
2. Do not add third-party data or archival materials without separately checking
   their current terms.
3. The MIT license applies to this repository's original code and documentation,
   not to any data a user obtains separately.
4. Review generated `outputs/` before sharing: interaction logs can include
   provider responses, usage metadata, and cost estimates. They are ignored and
   omitted from `git archive` exports by default.
5. Users must obtain the data from the upstream archive themselves.

The repository's fresh Git history contains no raw/normalized data, runtime
outputs, or internal planning records.
