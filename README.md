# MRI Metadata Explorer

DICOM metadata extraction and visualization tool for brain MRI and PET studies. Built for the **AI for Health** coursework exercise: load DICOM files, extract metadata into a Pandas DataFrame, and generate summary statistics.

## Features

- Load DICOM folders (MRI, PET, or both combined)
- Extract modality, patient age, acquisition date, body part (with smart inference)
- Interactive **Streamlit** dashboard with Plotly charts
- Slice viewer with window/level controls
- MRI vs PET comparison views
- CSV / JSON export and paste-ready reports

## Quick start

```bash
git clone https://github.com/LikhitaYerra/MRI_METADATA_EXPLORER.git
cd MRI_METADATA_EXPLORER
pip install -r requirements.txt
streamlit run dicom_metadata_web.py
```

Open http://localhost:8501

## CLI usage

```bash
python dicom_metadata_explorer.py --folder /path/to/dicom/folder --output metadata.csv
```

## Data folders

Place your DICOM files in folders such as:

- `BrainTumorMRI/` — MR slices (`.dcm`)
- `BrainTumorPET/` — PET slices (`.dcm`)

In the web app sidebar, use **Preset → MRI + PET (both)** or enter a custom folder path.

Default paths (macOS):

- MRI: `~/Downloads/data/BrainTumorMRI`
- PET: `~/Downloads/data/BrainTumorPET`

## Project structure

| File | Description |
|------|-------------|
| `dicom_metadata_web.py` | Streamlit web app |
| `dicom_explorer_core.py` | Metadata extraction and analytics |
| `dicom_viz.py` | Plotly visualization helpers |
| `dicom_metadata_explorer.py` | Command-line tool |
| `DICOM_Metadata_Explorer.ipynb` | Notebook version |

## Example output

```
Total files: 106
MRI: 24 files (MR, 2007-07-20, BRAIN)
PET: 82 files (PT, 2007-08-03, BRAIN)
```

## Author

Likhita Yerra — [GitHub](https://github.com/LikhitaYerra)
