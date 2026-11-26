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

The first prepared subset uses MMA and conversion. The experimental unit is an independent reaction screening (not a row or NMR spectrum). Source rows from a reaction must stay in one fold. Preserve the original file and row provenance.

## Preparation

The pinned release inventory was checked: 15 MMA CSV files and Summary_for_folders.xlsx, with 16 rows of reaction metadata. The CSV for 80mma0.04 is absent. mapping.mma-v1.json matches all available MMA CSV headers and hashes and the workbook. It declares provisional automated screening, not expert-reviewed chemistry. The platform's documented Status=Timesweep identifies kinetic rows and tres is in minutes (https://doi.org/10.1039/D2DD00035K, SI Table S14).

```bash
python -m polysearch.data.monash_etl --mapping data/monash/mapping.mma-v1.json
```

The output is written under data/processed/monash-v1. It is ignored by Git. See [DATA](../../docs/DATA.md). No processed observations should be called real until the acquisition and mapping have actually passed.

GitHub Actions downloads the focused MMA subset and runs the provisional ETL when source or mapping code changes. Download the resulting artifact from the workflow run. Manual dispatch is also supported; unrelated pushes do not repeat the download.
