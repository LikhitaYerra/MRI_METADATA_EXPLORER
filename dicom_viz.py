"""Plotly visualization helpers for DICOM metadata explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def chart_modality_donut(summary: dict) -> go.Figure:
    df = pd.DataFrame({"modality": list(summary["modality_counts"]), "count": list(summary["modality_counts"].values())})
    fig = px.pie(df, names="modality", values="count", hole=0.55, color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(title="Modality distribution", height=360, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_body_part_bar(summary: dict) -> go.Figure:
    df = pd.DataFrame({"body_part": list(summary["body_part_counts"]), "count": list(summary["body_part_counts"].values())})
    fig = px.bar(df, x="body_part", y="count", color="body_part", title="Body region counts")
    fig.update_layout(showlegend=False, height=360, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_completeness_heatmap(q: pd.DataFrame, top_n: int = 16) -> go.Figure:
    subset = q.sort_values("completeness_pct", ascending=False).head(top_n)
    z = subset["completeness_pct"].tolist()
    fig = go.Figure(
        data=go.Heatmap(
            z=[z],
            x=subset["field"].tolist(),
            y=["Completeness %"],
            colorscale="Viridis",
            text=[[f"{v:.0f}%" for v in z]],
            texttemplate="%{text}",
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(title="Metadata completeness heatmap", height=280, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_completeness_radial(q: pd.DataFrame) -> go.Figure:
    top = q.sort_values("completeness_pct", ascending=False).head(8)
    fig = go.Figure(
        data=go.Scatterpolar(
            r=top["completeness_pct"],
            theta=top["field"],
            fill="toself",
            name="Completeness",
            line_color="#6366f1",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")),
        title="Top field completeness (radar)",
        height=420,
        margin=dict(t=50, b=20, l=40, r=40),
    )
    return fig


def chart_slice_timeline(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["slice_index"] = range(1, len(plot_df) + 1)
    if "instance_number" in plot_df.columns:
        plot_df["instance_num"] = pd.to_numeric(plot_df["instance_number"], errors="coerce")
    else:
        plot_df["instance_num"] = plot_df["slice_index"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=plot_df["slice_index"], y=plot_df["instance_num"], mode="lines+markers", name="Instance #", line=dict(color="#22d3ee")),
        secondary_y=False,
    )
    if "slice_thickness" in plot_df.columns:
        thickness = pd.to_numeric(plot_df["slice_thickness"], errors="coerce")
        fig.add_trace(
            go.Bar(x=plot_df["slice_index"], y=thickness, name="Slice thickness (mm)", marker_color="#a78bfa", opacity=0.5),
            secondary_y=True,
        )
    fig.update_layout(title="Slice stack timeline", height=360, margin=dict(t=50, b=20, l=20, r=20), legend=dict(orientation="h"))
    fig.update_xaxes(title_text="Slice index")
    fig.update_yaxes(title_text="Instance number", secondary_y=False)
    fig.update_yaxes(title_text="Thickness (mm)", secondary_y=True)
    return fig


def chart_image_dimensions(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["rows_num"] = pd.to_numeric(plot_df.get("rows"), errors="coerce")
    plot_df["cols_num"] = pd.to_numeric(plot_df.get("columns"), errors="coerce")
    plot_df = plot_df.dropna(subset=["rows_num", "cols_num"])
    fig = px.scatter(
        plot_df,
        x="cols_num",
        y="rows_num",
        color="modality",
        hover_name="file_name",
        title="Image matrix size per slice",
        labels={"cols_num": "Columns (px)", "rows_num": "Rows (px)"},
    )
    fig.update_layout(height=360, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_manufacturer_treemap(df: pd.DataFrame) -> go.Figure:
    if "manufacturer" not in df.columns:
        return go.Figure()
    plot_df = df.groupby(["manufacturer", "modality", "body_part_inferred"], dropna=False).size().reset_index(name="count")
    fig = px.treemap(plot_df, path=["manufacturer", "modality", "body_part_inferred"], values="count", title="Manufacturer → modality → body part")
    fig.update_layout(height=420, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_combined_dataset_bar(df: pd.DataFrame) -> go.Figure:
    if "dataset" not in df.columns:
        return go.Figure()
    counts = df["dataset"].value_counts().reset_index()
    counts.columns = ["dataset", "files"]
    fig = px.bar(
        counts,
        x="dataset",
        y="files",
        color="dataset",
        title="Files per dataset (MRI + PET)",
        color_discrete_map={"MRI": "#22d3ee", "PET": "#f472b6"},
    )
    fig.update_layout(showlegend=False, height=320, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_combined_modality_sunburst(df: pd.DataFrame) -> go.Figure:
    if "dataset" not in df.columns:
        return go.Figure()
    plot_df = df.groupby(["dataset", "modality"], dropna=False).size().reset_index(name="count")
    fig = px.sunburst(plot_df, path=["dataset", "modality"], values="count", title="Dataset → modality")
    fig.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_mri_pet_comparison(df_mri: pd.DataFrame, df_pet: pd.DataFrame) -> go.Figure:
    rows = [
        {"dataset": "MRI", "metric": "Files", "value": len(df_mri)},
        {"dataset": "PET", "metric": "Files", "value": len(df_pet)},
        {"dataset": "MRI", "metric": "Series", "value": df_mri["series_instance_uid"].nunique()},
        {"dataset": "PET", "metric": "Series", "value": df_pet["series_instance_uid"].nunique()},
    ]
    plot_df = pd.DataFrame(rows)
    fig = px.bar(plot_df, x="metric", y="value", color="dataset", barmode="group", title="MRI vs PET comparison", color_discrete_map={"MRI": "#22d3ee", "PET": "#f472b6"})
    fig.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_mri_pet_modality_pies(df_mri: pd.DataFrame, df_pet: pd.DataFrame) -> go.Figure:
    fig = make_subplots(1, 2, specs=[[{"type": "pie"}, {"type": "pie"}]], subplot_titles=("MRI modalities", "PET modalities"))
    mri_counts = df_mri["modality"].value_counts()
    pet_counts = df_pet["modality"].value_counts()
    fig.add_trace(go.Pie(labels=mri_counts.index, values=mri_counts.values, hole=0.45, marker_colors=["#22d3ee"]), 1, 1)
    fig.add_trace(go.Pie(labels=pet_counts.index, values=pet_counts.values, hole=0.45, marker_colors=["#f472b6"]), 1, 2)
    fig.update_layout(title="Modality breakdown", height=380, margin=dict(t=60, b=20, l=20, r=20))
    return fig


def chart_scan_dates(df: pd.DataFrame) -> go.Figure:
    if "acquisition_date_fmt" not in df.columns:
        return go.Figure()
    counts = df["acquisition_date_fmt"].value_counts().reset_index()
    counts.columns = ["date", "slices"]
    fig = px.area(counts, x="date", y="slices", title="Slices per scan date", markers=True)
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20))
    return fig


def chart_pixel_histogram(path: str) -> go.Figure:
    import pydicom

    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(float)
    fig = px.histogram(x=arr.flatten(), nbins=80, title="Pixel intensity histogram", labels={"x": "Intensity", "y": "Count"})
    fig.update_traces(marker_color="#818cf8")
    fig.update_layout(height=300, margin=dict(t=50, b=20, l=20, r=20), showlegend=False)
    return fig


def chart_overall_score(q: pd.DataFrame) -> go.Figure:
    score = round(q["completeness_pct"].mean(), 1)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Average metadata completeness"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#6366f1"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 70], "color": "#fef3c7"},
                    {"range": [70, 100], "color": "#d1fae5"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=30, r=30))
    return fig
