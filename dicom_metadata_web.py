"""Advanced DICOM Metadata Explorer — Streamlit web application."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dicom_viz import (
    chart_body_part_bar,
    chart_combined_dataset_bar,
    chart_combined_modality_sunburst,
    chart_completeness_heatmap,
    chart_completeness_radial,
    chart_image_dimensions,
    chart_manufacturer_treemap,
    chart_modality_donut,
    chart_mri_pet_comparison,
    chart_mri_pet_modality_pies,
    chart_overall_score,
    chart_pixel_histogram,
    chart_scan_dates,
    chart_slice_timeline,
)

from dicom_explorer_core import (
    assignment_table,
    build_montage,
    compare_summaries,
    completeness_report,
    load_dicom_folder,
    load_dicom_image,
    load_mri_pet_pair,
    read_all_dicom_tags,
    series_summary,
    summarize_metadata,
)

PRESETS = {
    "MRI + PET (both)": None,
    "Brain Tumor MRI": "/Users/likhitayerra/Downloads/data/BrainTumorMRI",
    "Brain Tumor PET": "/Users/likhitayerra/Downloads/data/BrainTumorPET",
    "Sample DICOM (demo)": "/Users/likhitayerra/health/data/dicom",
}
MRI_PATH = Path(PRESETS["Brain Tumor MRI"])
PET_PATH = Path(PRESETS["Brain Tumor PET"])

st.set_page_config(
    page_title="DICOM Metadata Explorer Pro",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }
    .hero {
        background: radial-gradient(circle at top left, #0891b2, #4f46e5 45%, #7c3aed 100%);
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(79,70,229,0.25);
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2rem; font-weight: 700; }
    .hero p { color: #e0e7ff; margin: 0.4rem 0 0 0; font-size: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def scan_mri_pet() -> None:
    if not MRI_PATH.is_dir() or not PET_PATH.is_dir():
        raise FileNotFoundError("BrainTumorMRI or BrainTumorPET folder not found.")
    mri, pet, combined = load_mri_pet_pair(MRI_PATH, PET_PATH)
    st.session_state["df_mri"] = mri
    st.session_state["df_pet"] = pet
    st.session_state["df"] = combined
    st.session_state["source_name"] = "MRI + PET combined"
    st.session_state["source_path"] = f"{MRI_PATH} + {PET_PATH}"
    st.session_state["compare_loaded"] = True


def clear_compare_state() -> None:
    st.session_state.pop("df_mri", None)
    st.session_state.pop("df_pet", None)
    st.session_state.pop("compare_loaded", None)


def scan_folder(path: Path, label: str) -> None:
    clear_compare_state()
    st.session_state["df"] = load_dicom_folder(path)
    st.session_state["source_name"] = label
    st.session_state["source_path"] = str(path)


def _write_uploaded_files(files, tmp: Path) -> int:
    for file in files:
        dest = tmp / file.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file.getvalue())
    return len(files)


def _extract_zip_to_dir(zip_file, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_file.getvalue())) as zf:
        for member in zf.infolist():
            if member.is_dir() or member.filename.endswith("/"):
                continue
            target = dest / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member.filename))


def scan_upload(files) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="dicom_upload_"))
    count = _write_uploaded_files(files, tmp)
    clear_compare_state()
    st.session_state["df"] = load_dicom_folder(tmp)
    st.session_state["source_name"] = f"Upload ({count} files)"
    st.session_state["source_path"] = str(tmp)


def scan_upload_folder_zip(zip_file) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="dicom_folder_"))
    _extract_zip_to_dir(zip_file, tmp)
    clear_compare_state()
    df = load_dicom_folder(tmp)
    if df.empty:
        raise ValueError("No DICOM files found in the uploaded folder.")
    st.session_state["df"] = df
    st.session_state["source_name"] = f"Folder upload ({zip_file.name})"
    st.session_state["source_path"] = str(tmp)


def scan_upload_mri_pet_zips(mri_zip, pet_zip) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="dicom_mri_pet_"))
    mri_dir = tmp / "mri"
    pet_dir = tmp / "pet"
    _extract_zip_to_dir(mri_zip, mri_dir)
    _extract_zip_to_dir(pet_zip, pet_dir)
    mri, pet, combined = load_mri_pet_pair(mri_dir, pet_dir)
    if mri.empty and pet.empty:
        raise ValueError("No DICOM files found in the uploaded MRI or PET folders.")
    st.session_state["df_mri"] = mri
    st.session_state["df_pet"] = pet
    st.session_state["df"] = combined
    st.session_state["source_name"] = "MRI + PET folder upload"
    st.session_state["source_path"] = f"{mri_zip.name} + {pet_zip.name}"
    st.session_state["compare_loaded"] = True


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out.columns and st.session_state.get("filter_dataset", "All") != "All":
        out = out[out["dataset"] == st.session_state["filter_dataset"]]
    if st.session_state.get("filter_modality", "All") != "All":
        out = out[out["modality"] == st.session_state["filter_modality"]]
    if search := st.session_state.get("filter_search", "").strip():
        mask = out.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
        out = out[mask]
    return out


if "df" not in st.session_state and MRI_PATH.is_dir() and PET_PATH.is_dir():
    try:
        scan_mri_pet()
    except Exception:
        pass


st.markdown(
    """
    <div class="hero">
      <h1>DICOM Metadata Explorer Pro</h1>
      <p>Assignment-ready metadata extraction, smart body-part inference, window/level viewer, MRI vs PET compare.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Load data")
    source = st.radio(
        "Source",
        ["Preset", "Folder path", "Upload files", "Upload folder"],
        label_visibility="collapsed",
    )

    uploads = None
    folder_zip = None
    mri_zip = None
    pet_zip = None
    path = None
    preset = None

    if source == "Preset":
        preset = st.selectbox("Dataset", list(PRESETS.keys()), index=0)
        if preset == "MRI + PET (both)":
            path = MRI_PATH
            st.caption(f"MRI: {MRI_PATH}")
            st.caption(f"PET: {PET_PATH}")
        else:
            path = Path(PRESETS[preset])
            st.caption(path)
    elif source == "Folder path":
        path = Path(st.text_input("Path", PRESETS["Brain Tumor MRI"]))
    elif source == "Upload files":
        uploads = st.file_uploader(
            "Select .dcm files",
            type=["dcm", "dicom"],
            accept_multiple_files=True,
            key="upload_dicom_files",
        )
        st.caption("Tip: on Mac, select all files inside the folder in Finder (Cmd+A).")
    else:
        folder_mode = st.radio(
            "Folder upload type",
            ["Single DICOM folder", "MRI + PET (2 folders)"],
            horizontal=True,
            key="upload_folder_mode",
        )
        if folder_mode == "Single DICOM folder":
            folder_zip = st.file_uploader(
                "Upload folder as ZIP",
                type=["zip"],
                key="upload_folder_zip",
            )
            st.caption("Right-click your DICOM folder → **Compress** (Mac) or **Send to → Compressed folder** (Windows).")
        else:
            mri_zip = st.file_uploader("MRI folder (ZIP)", type=["zip"], key="upload_mri_zip")
            pet_zip = st.file_uploader("PET folder (ZIP)", type=["zip"], key="upload_pet_zip")
            st.caption("Zip each study folder separately, then upload both ZIP files.")

    c1, c2 = st.columns(2)
    if c1.button("Scan", type="primary", use_container_width=True):
        try:
            if source == "Upload files":
                if uploads:
                    scan_upload(uploads)
                else:
                    st.error("Upload files first.")
            elif source == "Upload folder":
                if folder_mode == "Single DICOM folder":
                    if folder_zip:
                        scan_upload_folder_zip(folder_zip)
                    else:
                        st.error("Upload a ZIP of your DICOM folder first.")
                elif mri_zip and pet_zip:
                    scan_upload_mri_pet_zips(mri_zip, pet_zip)
                else:
                    st.error("Upload both MRI and PET folder ZIP files.")
            elif source == "Preset" and preset == "MRI + PET (both)":
                scan_mri_pet()
            elif path is not None and path.is_dir():
                scan_folder(path, str(path))
            else:
                st.error("Folder not found.")
        except Exception as exc:
            st.error(str(exc))

    if c2.button("Reload MRI+PET", use_container_width=True):
        try:
            scan_mri_pet()
        except Exception as exc:
            st.error(str(exc))

    st.divider()
    st.header("Filters")
    base_df = st.session_state.get("df", pd.DataFrame())
    if "dataset" in base_df.columns:
        datasets = ["All"] + sorted(base_df["dataset"].dropna().unique().tolist())
        st.session_state["filter_dataset"] = st.selectbox("Dataset", datasets)
    mods = ["All"] + sorted(base_df["modality"].dropna().unique().tolist()) if not base_df.empty else ["All"]
    st.session_state["filter_modality"] = st.selectbox("Modality", mods)
    st.session_state["filter_search"] = st.text_input("Search", placeholder="SIEMENS, t2, brain...")

df = st.session_state.get("df", pd.DataFrame())
if df.empty:
    st.info(
        "Load data in the sidebar: **Upload folder** (ZIP) works on Streamlit Cloud, "
        "or use **Upload files** / **Folder path** locally. Then click **Scan**."
    )
    st.stop()

df = filter_df(df)
summary = summarize_metadata(df)
assign_df = assignment_table(df)
df_mri = st.session_state.get("df_mri")
df_pet = st.session_state.get("df_pet")
both_loaded = df_mri is not None and df_pet is not None

tabs = st.tabs(
    ["Overview", "Visualizations", "Assignment table", "Explorer", "Viewer", "Compare MRI/PET", "Quality", "Export"]
)

with tabs[0]:
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total files", summary["total_files"])
    if both_loaded:
        m2.metric("MRI slices", len(df_mri))
        m3.metric("PET slices", len(df_pet))
    else:
        m2.metric("Studies", summary.get("unique_studies", 0))
        m3.metric("Series", summary.get("unique_series", 0))
    m4.metric("Modalities", len(summary["modality_counts"]))
    m5.metric("Age tagged", summary["files_with_patient_age"])
    m6.metric("Scan dates", summary["unique_acquisition_dates"])

    if both_loaded:
        st.plotly_chart(chart_combined_dataset_bar(st.session_state["df"]), use_container_width=True, key="overview_combined_bar")
        st.plotly_chart(chart_combined_modality_sunburst(st.session_state["df"]), use_container_width=True, key="overview_combined_sunburst")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.plotly_chart(chart_modality_donut(summary), use_container_width=True, key="overview_modality_donut")
    with r1c2:
        st.plotly_chart(chart_body_part_bar(summary), use_container_width=True, key="overview_body_part_bar")
    with r1c3:
        q_preview = completeness_report(df)
        st.plotly_chart(chart_overall_score(q_preview), use_container_width=True, key="overview_completeness_gauge")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(chart_slice_timeline(df), use_container_width=True, key="overview_slice_timeline")
    with r2c2:
        st.plotly_chart(chart_scan_dates(df), use_container_width=True, key="overview_scan_dates")

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(chart_image_dimensions(df), use_container_width=True, key="overview_image_dims")
    with r3c2:
        st.plotly_chart(chart_manufacturer_treemap(df), use_container_width=True, key="overview_treemap")

    st.success(
        f"Loaded **{summary['total_files']}** files from `{st.session_state.get('source_name', 'dataset')}`. "
        f"Inferred body part filled for **{int(df['body_part_inferred'].ne('Unknown').sum())}/{len(df)}** slices."
    )

with tabs[1]:
    st.markdown("### Interactive visualization dashboard")
    q = completeness_report(df)

    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(chart_completeness_heatmap(q), use_container_width=True, key="viz_completeness_heatmap")
        st.plotly_chart(chart_modality_donut(summary), use_container_width=True, key="viz_modality_donut")
    with v2:
        st.plotly_chart(chart_completeness_radial(q), use_container_width=True, key="viz_completeness_radial")
        st.plotly_chart(chart_body_part_bar(summary), use_container_width=True, key="viz_body_part_bar")

    v3, v4 = st.columns(2)
    with v3:
        st.plotly_chart(chart_slice_timeline(df), use_container_width=True, key="viz_slice_timeline")
    with v4:
        st.plotly_chart(chart_image_dimensions(df), use_container_width=True, key="viz_image_dims")

    st.plotly_chart(chart_manufacturer_treemap(df), use_container_width=True, key="viz_treemap")

    st.markdown("### Slice gallery")
    if both_loaded:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**MRI thumbnails**")
            mcols = st.columns(3)
            for i, (_, row) in enumerate(df_mri.head(6).iterrows()):
                with mcols[i % 3]:
                    try:
                        img, _, _, _ = load_dicom_image(row["file_path"])
                        st.image(img, caption=row["file_name"], use_container_width=True)
                    except Exception:
                        st.caption(row["file_name"])
        with g2:
            st.markdown("**PET thumbnails**")
            pcols = st.columns(3)
            for i, (_, row) in enumerate(df_pet.head(6).iterrows()):
                with pcols[i % 3]:
                    try:
                        img, _, _, _ = load_dicom_image(row["file_path"])
                        st.image(img, caption=row["file_name"], use_container_width=True)
                    except Exception:
                        st.caption(row["file_name"])
    else:
        gallery_n = st.slider("Number of thumbnails", 4, min(24, len(df)), 12, key="viz_gallery_n")
        cols = st.columns(4)
        for i, (_, row) in enumerate(df.head(gallery_n).iterrows()):
            with cols[i % 4]:
                try:
                    img, _, _, _ = load_dicom_image(row["file_path"])
                    st.image(img, caption=row["file_name"], use_container_width=True)
                except Exception:
                    st.caption(row["file_name"])

with tabs[2]:
    st.markdown("### Coursework columns (cleaned for submission)")
    st.dataframe(assign_df, use_container_width=True, height=360)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(chart_modality_donut(summary), use_container_width=True, key="assign_modality_donut")
    with c2:
        st.plotly_chart(chart_body_part_bar(summary), use_container_width=True, key="assign_body_part_bar")
    with c3:
        st.plotly_chart(chart_scan_dates(df), use_container_width=True, key="assign_scan_dates")

with tabs[3]:
    show_cols = [
        "dataset", "file_name", "modality", "patient_age_parsed", "acquisition_date_fmt",
        "body_part_examined", "body_part_inferred", "study_description",
        "series_description", "manufacturer", "magnetic_field_strength",
        "slice_thickness", "instance_number",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, height=380)

    pick = st.selectbox("Full DICOM tag browser", df["file_name"].tolist(), key="explorer_file_pick")
    tag_path = df.loc[df["file_name"] == pick, "file_path"].iloc[0]
    st.dataframe(read_all_dicom_tags(tag_path), use_container_width=True, height=360)

with tabs[4]:
    if both_loaded:
        view_set = st.radio("View slices from", ["MRI", "PET", "All combined"], horizontal=True, key="viewer_dataset")
        if view_set == "MRI":
            view_df = df_mri
        elif view_set == "PET":
            view_df = df_pet
        else:
            view_df = st.session_state["df"]
    else:
        view_df = df

    sorted_names = view_df["file_name"].tolist()
    idx = st.slider("Slice", 0, max(len(sorted_names) - 1, 0), 0, key="viewer_slice_idx")
    chosen = sorted_names[idx]
    path = view_df.loc[view_df["file_name"] == chosen, "file_path"].iloc[0]

    try:
        _, ds, default_c, default_w = load_dicom_image(path)
        wc = st.slider("Window center", float(default_c - 1000), float(default_c + 1000), float(default_c), key="viewer_window_center")
        ww = st.slider("Window width", 1.0, float(max(default_w * 4, 500)), float(default_w), key="viewer_window_width")
        img, _, _, _ = load_dicom_image(path, wc, ww)
    except Exception as exc:
        st.error(str(exc))
        img = None

    v1, v2, v3 = st.columns([2, 1, 1])
    with v1:
        if img is not None:
            st.image(img, caption=f"{chosen} · slice {idx + 1}/{len(sorted_names)}", use_container_width=True)
            st.plotly_chart(chart_pixel_histogram(path), use_container_width=True, key="viewer_pixel_histogram")
    with v2:
        row = view_df[view_df["file_name"] == chosen].iloc[0]
        st.markdown("**Slice info**")
        for k in ["modality", "series_description", "body_part_inferred", "rows", "columns", "slice_thickness", "magnetic_field_strength"]:
            st.write(f"**{k.replace('_',' ').title()}:** {row.get(k, '—')}")
    with v3:
        st.markdown("**Montage**")
        try:
            montage = build_montage(view_df, max_images=9)
            if montage is not None:
                st.image(montage, caption="First 9 slices", use_container_width=True)
        except Exception as exc:
            st.caption(f"Montage unavailable: {exc}")

with tabs[5]:
    if not both_loaded:
        st.warning("Reload **MRI+PET** in the sidebar to compare both datasets.")
    else:
        st.plotly_chart(chart_mri_pet_comparison(df_mri, df_pet), use_container_width=True, key="compare_bar")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_mri_pet_modality_pies(df_mri, df_pet), use_container_width=True, key="compare_pies")
        with c2:
            st.plotly_chart(chart_scan_dates(df_mri), use_container_width=True, key="compare_mri_dates")
            st.caption("MRI scan dates")
            st.plotly_chart(chart_scan_dates(df_pet), use_container_width=True, key="compare_pet_dates")
            st.caption("PET scan dates")
        st.dataframe(compare_summaries(df_mri, "MRI", df_pet, "PET"), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**MRI**")
            st.dataframe(assignment_table(df_mri).head(8), use_container_width=True)
        with c2:
            st.markdown("**PET**")
            st.dataframe(assignment_table(df_pet).head(8), use_container_width=True)

with tabs[6]:
    q = completeness_report(df)
    st.dataframe(q, use_container_width=True, height=280)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_completeness_heatmap(q), use_container_width=True, key="quality_heatmap")
    with c2:
        st.plotly_chart(chart_completeness_radial(q), use_container_width=True, key="quality_radial")
    st.plotly_chart(
        px.bar(q.sort_values("completeness_pct"), x="completeness_pct", y="field", orientation="h",
               title="Metadata completeness (%)", color="completeness_pct", color_continuous_scale="Teal"),
        use_container_width=True,
        key="quality_bar",
    )

with tabs[7]:
    st.download_button(
        "Download assignment CSV",
        assign_df.to_csv(index=False).encode("utf-8"),
        "dicom_assignment_table.csv",
        use_container_width=True,
        key="export_assignment_csv",
    )
    st.download_button(
        "Download full metadata CSV",
        df.drop(columns=["file_path"], errors="ignore").to_csv(index=False).encode("utf-8"),
        "dicom_full_metadata.csv",
        use_container_width=True,
        key="export_full_csv",
    )
    st.subheader("Paste-ready report")
    mri_line = f"MRI files: {len(df_mri)} (modality MR, date {df_mri['acquisition_date_fmt'].iloc[0]})" if both_loaded else ""
    pet_line = f"PET files: {len(df_pet)} (modality PT, date {df_pet['acquisition_date_fmt'].iloc[0]})" if both_loaded else ""
    st.code(
        f"""DICOM Metadata Explorer — Results

Source: {st.session_state.get('source_name', 'dataset')}
Total files: {summary['total_files']}
{mri_line}
{pet_line}
Studies: {summary.get('unique_studies', 0)} | Series: {summary.get('unique_series', 0)}

Modality counts:
{chr(10).join(f'  - {k}: {v}' for k, v in summary['modality_counts'].items())}

Body part examined (inferred when tag missing):
{chr(10).join(f'  - {k}: {v}' for k, v in summary['body_part_counts'].items())}

Patient age present: {summary['files_with_patient_age']}/{summary['total_files']}
Acquisition date present: {summary['files_with_acquisition_date']}/{summary['total_files']}
Unique acquisition dates: {summary['unique_acquisition_dates']}""",
        language="text",
    )
