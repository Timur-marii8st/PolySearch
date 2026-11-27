# RAFT Knowledge Base — independent source

Ringleb, Köster, Zechel, Schubert, *Unveiling the knowledge of a RAFT polymerization database obtained from an automated parallel synthesizer*, Digital Discovery (2025), DOI https://doi.org/10.1039/D5DD00177C.

Pinned curated source/software archive: Köster and Ringleb, https://doi.org/10.5281/zenodo.15230215, version v1.0.0, CC BY 4.0 (Zenodo record). MD5 e7dc1c91cee856b8cc286fae7cc78d5f, approximately 5.1 MB. The separate raw spectroscopy archive is https://doi.org/10.5281/zenodo.14018513 (~3.3 GB) and is not downloaded by this pipeline. Website https://raft-knowledgebase.de/. Upstream repository https://github.com/Bizbalt/RAFT-knowledgebase.

Published study describes 539 experiments and 234 curated successful kinetics. Exact numbers in the pinned files must be measured, not inferred from the publication. RAFT is a different chemistry and protocol from Monash chain transfer flow polymerization. Do not equate time, agents, solvent, temperature or molecular-weight methods without explicit reconciliation.

Download and inventory only:
```bash
python -m pip install openpyxl==3.1.5
python -m polysearch.data.raft_source
```

Source archive and raw workbooks are excluded from Git. Keep observed measurements and fitted/interpolated curves distinct. No copying upstream software into PolySearch, no missing molecular weights imputed from predictions. Review units, timepoint alignment, replicate grouping and license before commercial use.

## Conservative observed-label extraction

`python -m polysearch.data.raft_etl` reads only the pinned evaluation workbook (SHA256 9c40f72053e66a7a6c4bdd5d14a92e2af1eb9a89470dc1045824c6f6f8741060). Explicit positive author AI flags required, explicit negative flags override. Uses real numeric cached observations, discards nonphysical/out-of-calibration and flagged SEC values; retains missing labels as blanks. No fitted or interpolated labels, no reference t=0 pseudo-observations. Nominal hours, not the corrected actual sampling times; preserves original sample IDs and workbook row/column coordinates. Reaction codes are preserved, not treated as ordered numerical chemistry features. Group repeats by monomer/RAFT agent/solvent. This conservative subset is NOT the published 234-curve curation and cannot be directly mixed with Monash. No cross-source model fitted yet.
