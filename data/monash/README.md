# Monash v1 — source metadata

Pinned release: https://doi.org/10.26180/29208134.v1 (Figshare article 29208134, version 1).

Asghar, Amna; Zhang, Bo; Junkers, Tanja; Jafari, Vianna (2025).
*A Data-Driven approach to the Generalization of Free Radical Polymerization Kinetic Models via Automated Flow Chemistry - Repository*.
Monash University. CC BY 4.0.
Publication: https://doi.org/10.1039/D5RE00264H.

The public archive is downloaded into ignored local storage, not committed. Run:

```bash
python -m pip install openpyxl==3.1.5
python -m polysearch.data.monash_source
```

Output: pinned file manifest with checksums and raw schema inventory. The release provides multiple monomers, experiment-code summary workbook, CSV measurements and plots. No conversion values, column names or experiment-to-condition mappings are assumed before inspection.

The first prepared subset will use MMA and conversion. The experimental unit is an independent reaction screening (not a row or NMR spectrum). Source rows from a reaction must stay in one fold. Preserve the original file and row provenance.

## Preparation

Review inventory.json and source workbook. Copy mapping.template.json to mapping.reviewed.json only when exact file names, hashes, experiment codes, conditions, units and accepted timesweeps have been checked. Unreviewed mappings are rejected.

```bash
python -m polysearch.data.monash_etl --mapping data/monash/mapping.reviewed.json
```

The output is written under data/processed/monash-v1. It is ignored by Git. See [DATA](../../docs/DATA.md). No processed observations should be called real until the acquisition and mapping have actually passed.
