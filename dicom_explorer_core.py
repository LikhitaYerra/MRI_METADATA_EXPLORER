"""Extended DICOM metadata extraction and analytics."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

CORE_TAGS = [
    "Modality",
    "PatientAge",
    "PatientSex",
    "PatientID",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "BodyPartExamined",
    "StudyDescription",
    "SeriesDescription",
    "ProtocolName",
    "Manufacturer",
    "InstitutionName",
    "MagneticFieldStrength",
    "MRAcquisitionType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "WindowCenter",
    "WindowWidth",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "InstanceNumber",
    "ImageType",
    "PhotometricInterpretation",
]


def _value(ds: pydicom.Dataset, tag: str):
    val = getattr(ds, tag, None)
    if val in (None, ""):
        return None
    if isinstance(val, pydicom.multival.MultiValue):
        return "\\".join(str(v) for v in val)
    return str(val).strip()


def _tag_key(tag: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", tag)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _acquisition_date(ds: pydicom.Dataset) -> str | None:
    for tag in ("AcquisitionDate", "SeriesDate", "StudyDate", "ContentDate"):
        val = _value(ds, tag)
        if val:
            return val
    return None


def format_dicom_date(value: str | None) -> str | None:
    if not value or len(value) != 8 or not value.isdigit():
        return value
    return f"{value[0:4]}-{value[4:6]}-{value[6:8]}"


def parse_patient_age(value: str | None) -> str | None:
    if not value:
        return None
    match = re.match(r"^(\d+)([YMWD])$", value.strip())
    if not match:
        return value
    num, unit = match.groups()
    labels = {"Y": "years", "M": "months", "W": "weeks", "D": "days"}
    if num == "000" and unit == "Y":
        return "De-identified / unknown"
    return f"{int(num)} {labels[unit]}"


def infer_body_part(body_part: str | None, study_desc: str | None, series_desc: str | None) -> str:
    if body_part:
        return body_part
    text = " ".join(filter(None, [study_desc, series_desc])).lower()
    mapping = {
        "brain": "BRAIN",
        "crane": "BRAIN",
        "head": "HEAD",
        "chest": "CHEST",
        "lung": "CHEST",
        "abdomen": "ABDOMEN",
        "pelvis": "PELVIS",
        "spine": "SPINE",
        "heart": "HEART",
    }
    for key, label in mapping.items():
        if key in text:
            return label
    return "Unknown"


def extract_metadata(dicom_path: Path) -> dict:
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    row = {
        "file_name": dicom_path.name,
        "file_path": str(dicom_path),
    }
    for tag in CORE_TAGS:
        row[_tag_key(tag)] = _value(ds, tag)

    acq = _acquisition_date(ds)
    row["acquisition_date"] = acq
    row["acquisition_date_fmt"] = format_dicom_date(acq)
    row["patient_age_parsed"] = parse_patient_age(row.get("patient_age"))
    row["body_part_inferred"] = infer_body_part(
        row.get("body_part_examined"),
        row.get("study_description"),
        row.get("series_description"),
    )
    return row


def load_dicom_folder(folder: str | Path) -> pd.DataFrame:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    rows: list[dict] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".dcm", ".dicom", ""}:
            continue
        try:
            rows.append(extract_metadata(path))
        except (InvalidDicomError, PermissionError, OSError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "instance_number" in df.columns:
        df["_instance_num"] = pd.to_numeric(df["instance_number"], errors="coerce")
        df = df.sort_values(["series_instance_uid", "_instance_num", "file_name"], na_position="last")
        df = df.drop(columns=["_instance_num"])
    return df.reset_index(drop=True)


def summarize_metadata(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total_files": 0}

    body_col = "body_part_inferred" if "body_part_inferred" in df.columns else "body_part_examined"
    return {
        "total_files": len(df),
        "modality_counts": df["modality"].value_counts(dropna=False).to_dict(),
        "body_part_counts": df[body_col].value_counts(dropna=False).to_dict(),
        "files_with_patient_age": int(df["patient_age"].notna().sum()),
        "files_with_acquisition_date": int(df["acquisition_date"].notna().sum()),
        "unique_acquisition_dates": int(df["acquisition_date"].nunique(dropna=True)),
        "unique_studies": int(df["study_instance_uid"].nunique(dropna=True)) if "study_instance_uid" in df else 0,
        "unique_series": int(df["series_instance_uid"].nunique(dropna=True)) if "series_instance_uid" in df else 0,
        "manufacturers": df["manufacturer"].value_counts(dropna=False).to_dict() if "manufacturer" in df else {},
    }


def load_mri_pet_pair(mri_folder: str | Path, pet_folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load MRI and PET folders; return separate frames and a combined tagged frame."""
    mri = load_dicom_folder(mri_folder)
    pet = load_dicom_folder(pet_folder)
    mri = mri.copy()
    pet = pet.copy()
    mri["dataset"] = "MRI"
    pet["dataset"] = "PET"
    combined = pd.concat([mri, pet], ignore_index=True)
    return mri, pet, combined
    """Required coursework columns with cleaned values."""
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "file_name": df["file_name"],
            "modality": df["modality"],
            "patient_age": df.get("patient_age_parsed", df["patient_age"]),
            "acquisition_date": df.get("acquisition_date_fmt", df["acquisition_date"]),
            "body_part_examined": df.get("body_part_inferred", df["body_part_examined"]),
        }
    )


def series_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_cols = [
        "modality",
        "series_description",
        "study_description",
        "body_part_inferred",
        "acquisition_date_fmt",
        "magnetic_field_strength",
        "manufacturer",
    ]
    existing = [c for c in group_cols if c in df.columns]
    return (
        df.groupby(existing, dropna=False)
        .agg(
            slices=("file_name", "count"),
            rows=("rows", "first"),
            columns=("columns", "first"),
            slice_thickness_mm=("slice_thickness", "first"),
        )
        .reset_index()
        .sort_values("slices", ascending=False)
    )


def compare_summaries(df_a: pd.DataFrame, name_a: str, df_b: pd.DataFrame, name_b: str) -> pd.DataFrame:
    sa, sb = summarize_metadata(df_a), summarize_metadata(df_b)
    rows = []
    for label, key in [
        ("Total files", "total_files"),
        ("Studies", "unique_studies"),
        ("Series", "unique_series"),
        ("Unique scan dates", "unique_acquisition_dates"),
    ]:
        rows.append({"metric": label, name_a: sa.get(key, 0), name_b: sb.get(key, 0)})
    rows.append({
        "metric": "Modalities",
        name_a: ", ".join(f"{k}={v}" for k, v in sa.get("modality_counts", {}).items()),
        name_b: ", ".join(f"{k}={v}" for k, v in sb.get("modality_counts", {}).items()),
    })
    rows.append({
        "metric": "Manufacturers",
        name_a: ", ".join(f"{k}={v}" for k, v in sa.get("manufacturers", {}).items()),
        name_b: ", ".join(f"{k}={v}" for k, v in sb.get("manufacturers", {}).items()),
    })
    return pd.DataFrame(rows)


def completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    priority = [
        "modality",
        "patient_age",
        "acquisition_date",
        "body_part_examined",
        "body_part_inferred",
        "series_description",
        "manufacturer",
    ]
    cols = [c for c in priority if c in df.columns] + [
        c for c in df.columns if c not in {"file_path"} and c not in priority
    ]
    stats = []
    for col in cols:
        filled = int(df[col].notna().sum())
        stats.append(
            {
                "field": col,
                "filled": filled,
                "missing": len(df) - filled,
                "completeness_pct": round(100 * filled / len(df), 1),
            }
        )
    return pd.DataFrame(stats)


def read_all_dicom_tags(path: str | Path) -> pd.DataFrame:
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    rows = []
    for elem in ds.iterall():
        if elem.VR == "SQ":
            continue
        rows.append({"tag": str(elem.tag), "name": elem.name, "value": str(elem.value)[:200]})
    return pd.DataFrame(rows)


def _window_defaults(ds: pydicom.Dataset) -> tuple[float, float]:
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is not None and width is not None:
        if isinstance(center, pydicom.multival.MultiValue):
            center, width = float(center[0]), float(width[0])
        else:
            center, width = float(center), float(width)
        return center, max(width, 1.0)
    arr_preview = ds.pixel_array.astype(float)
    return float(np.median(arr_preview)), float(arr_preview.max() - arr_preview.min() or 1)


def load_dicom_image(path: str | Path, window_center: float | None = None, window_width: float | None = None):
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(float)
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    default_center, default_width = _window_defaults(ds)
    center = window_center if window_center is not None else default_center
    width = max(window_width if window_width is not None else default_width, 1.0)

    low = center - width / 2
    high = center + width / 2
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    rgb = (arr * 255).astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        rgb = 255 - rgb
    return rgb, ds, default_center, default_width


def build_montage(df: pd.DataFrame, max_images: int = 9) -> np.ndarray | None:
    if df.empty:
        return None
    subset = df.head(max_images)
    images = []
    for path in subset["file_path"]:
        img, _, _, _ = load_dicom_image(path)
        images.append(img)
    if not images:
        return None

    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    cols = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / cols))
    canvas = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        ih, iw = img.shape[:2]
        canvas[r * h : r * h + ih, c * w : c * w + iw] = img
    return canvas
