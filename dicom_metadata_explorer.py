"""DICOM Metadata Explorer — extract and summarize metadata from a folder of DICOM files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError


METADATA_FIELDS = {
    "file_name": None,
    "file_path": None,
    "modality": "Modality",
    "patient_age": "PatientAge",
    "acquisition_date": "AcquisitionDate",
    "body_part_examined": "BodyPartExamined",
}


def _get_tag(ds: pydicom.Dataset, tag_name: str) -> str | None:
    value = getattr(ds, tag_name, None)
    if value in (None, ""):
        return None
    return str(value).strip()


def _acquisition_date(ds: pydicom.Dataset) -> str | None:
    for tag in ("AcquisitionDate", "StudyDate", "SeriesDate"):
        value = _get_tag(ds, tag)
        if value:
            return value
    return None


def extract_metadata(dicom_path: Path) -> dict:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    return {
        "file_name": dicom_path.name,
        "file_path": str(dicom_path),
        "modality": _get_tag(ds, "Modality"),
        "patient_age": _get_tag(ds, "PatientAge"),
        "acquisition_date": _acquisition_date(ds),
        "body_part_examined": _get_tag(ds, "BodyPartExamined"),
    }


def load_dicom_folder(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[dict] = []
    errors: list[str] = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".dcm", ".dicom", ""}:
            continue
        try:
            rows.append(extract_metadata(path))
        except (InvalidDicomError, PermissionError, OSError) as exc:
            errors.append(f"{path.name}: {exc}")

    df = pd.DataFrame(rows)
    if errors:
        print(f"Skipped {len(errors)} non-DICOM or unreadable file(s).")
    return df


def summarize_metadata(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_files": 0}

    summary = {
        "total_files": len(df),
        "modality_counts": df["modality"].value_counts(dropna=False).to_dict(),
        "body_part_counts": df["body_part_examined"].value_counts(dropna=False).to_dict(),
        "files_with_patient_age": int(df["patient_age"].notna().sum()),
        "files_with_acquisition_date": int(df["acquisition_date"].notna().sum()),
        "unique_acquisition_dates": int(df["acquisition_date"].nunique(dropna=True)),
    }
    return summary


def print_report(df: pd.DataFrame, summary: dict) -> None:
    print("\n=== DICOM Metadata Table ===")
    print(df.to_string(index=False))

    print("\n=== Summary Statistics ===")
    print(f"Total DICOM files: {summary['total_files']}")
    print("\nModality counts:")
    for key, value in summary["modality_counts"].items():
        print(f"  {key}: {value}")

    print("\nBody part examined counts:")
    for key, value in summary["body_part_counts"].items():
        print(f"  {key}: {value}")

    print(f"\nFiles with patient age: {summary['files_with_patient_age']}")
    print(f"Files with acquisition date: {summary['files_with_acquisition_date']}")
    print(f"Unique acquisition dates: {summary['unique_acquisition_dates']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and summarize DICOM metadata.")
    parser.add_argument(
        "--folder",
        default="data/dicom",
        help="Path to folder containing DICOM files (searched recursively).",
    )
    parser.add_argument(
        "--output",
        default="data/dicom_metadata.csv",
        help="CSV path for extracted metadata.",
    )
    args = parser.parse_args()

    df = load_dicom_folder(args.folder)
    summary = summarize_metadata(df)
    print_report(df, summary)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nSaved metadata table to: {output}")


if __name__ == "__main__":
    main()
