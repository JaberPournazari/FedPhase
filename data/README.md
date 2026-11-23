# Data

## MIT-140 (data2)
Per-cell CSVs with engineered features (`Cell_<i>.csv`, i = 0..139; 27 inputs, target `QD`), derived from the public
MIT–Stanford–Toyota fast-charging battery dataset (Severson et al., *Nature Energy* 2019; Attia et al., *Nature* 2020).
Download: **<ADD ZENODO / GITHUB RELEASE URL>** → unzip to `data/data2/` (or set `DATA_DIR`).
Check: `ls data/data2/Cell_*.csv | wc -l` → 140.

## WMG-15 (data1)
15 LG INR21700 M50 cells (V2G SoC pre-conditioning study), collected at WMG. Files per cell:
`PCA_Selected_Features<i>.csv`, `CapacityInterpolation<i>.csv`.
**<ADD: availability statement / DOI, or "available on request from the data owners">**
The loader (`fedphase/task.py`) expects these two file types in `DATA_DIR`.
