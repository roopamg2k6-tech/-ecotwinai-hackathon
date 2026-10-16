
from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from feature_hub import (
    DEFAULT_ELECTRICITY_PRICE_PER_KWH,
    DEFAULT_GRID_KG_CO2_PER_KWH,
    ENVIRONMENTS,
    EcoTwinController,
    detect_energy_anomalies,
    generate_ai_recommendations,
    generate_synthetic_energy_scenarios,
    scenario_comparison,
)

st.set_page_config(
    page_title="EcoTwin AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- Styling ----------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #1b2d25;
        --muted: #617068;
        --forest: #174b3c;
        --leaf: #277454;
        --citrus: #d9ef79;
        --coral: #dc8066;
        --paper: #f3f6f2;
        --line: #dce5dc;
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background-color: var(--paper);
        background-image:
            linear-gradient(rgba(23, 75, 60, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(23, 75, 60, 0.025) 1px, transparent 1px);
        background-size: 36px 36px;
    }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1480px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] {
        background: #e8efe9;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: var(--forest); }
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        padding: 1.4rem 1.6rem;
        border-radius: 8px;
        border-left: 5px solid var(--citrus);
        background-color: var(--forest);
        background-image:
            linear-gradient(90deg, rgba(15, 52, 41, 0.96) 0%, rgba(20, 70, 53, 0.82) 58%, rgba(18, 62, 47, 0.54) 100%),
            url('https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1800&q=85');
        background-position: center 54%;
        background-size: cover;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(23, 75, 60, 0.12);
        animation: rise-in 560ms ease-out both;
    }
    .hero-kicker {
        color: var(--citrus);
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .hero h1 {
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.25rem;
        line-height: 1.1;
        margin: 0 0 0.35rem;
    }
    .hero p { color: #e1eee6; margin: 0; }
    .hero-status {
        flex: 0 0 auto;
        border-left: 1px solid rgba(255, 255, 255, 0.3);
        padding-left: 1.25rem;
        color: #e1eee6;
        font-size: 0.8rem;
        line-height: 1.6;
    }
    .hero-status strong { display: block; color: white; font-size: 0.95rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    h2 { font-size: 1.45rem; }
    h3, h4 { font-size: 1.1rem; }
    .small-note { font-size: 0.85rem; color: var(--muted); }
    div[data-testid="stMetric"] {
        min-height: 112px;
        padding: 1rem 1.1rem;
        background: #fff;
        border: 1px solid var(--line);
        border-top: 3px solid var(--leaf);
        border-radius: 8px;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-weight: 600; }
    div[data-testid="stMetricValue"] {
        color: var(--forest);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        padding: 0.35rem;
        background: #e7ede7;
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.7rem;
        padding: 0 1rem;
        color: var(--muted);
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--forest);
        color: white;
    }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--citrus); }
    div.stButton > button, div[data-testid="stDownloadButton"] button {
        border-radius: 6px;
        border: 1px solid var(--forest);
        color: var(--forest);
        font-weight: 600;
        transition: background-color 140ms ease, color 140ms ease, border-color 140ms ease;
    }
    div.stButton > button:hover, div[data-testid="stDownloadButton"] button:hover {
        border-color: var(--leaf);
        color: var(--forest);
        background: #edf5e8;
    }
    div.stButton > button[kind="primary"] {
        background: var(--forest);
        color: white;
    }
    div[data-testid="stAlert"] { border-radius: 8px; }
    div[data-testid="stDataFrame"] {
        overflow: hidden;
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    div[data-testid="stPlotlyChart"] {
        padding: 0.4rem;
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .visual-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.4rem 0 1.2rem;
    }
    .energy-visual {
        position: relative;
        height: 154px;
        overflow: hidden;
        border: 1px solid rgba(23, 75, 60, 0.12);
        border-radius: 8px;
        background: var(--forest);
        isolation: isolate;
        animation: rise-in 650ms ease-out both;
    }
    .energy-visual:nth-child(2) { animation-delay: 100ms; }
    .energy-visual:nth-child(3) { animation-delay: 200ms; }
    .energy-visual img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 500ms ease;
    }
    .energy-visual:hover img { transform: scale(1.045); }
    .energy-visual::after {
        position: absolute;
        inset: 0;
        z-index: 0;
        background: linear-gradient(180deg, rgba(10, 28, 23, 0.02) 5%, rgba(10, 28, 23, 0.82) 100%);
        content: "";
    }
    .visual-copy {
        position: absolute;
        right: 1rem;
        bottom: 0.9rem;
        left: 1rem;
        z-index: 1;
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: white;
    }
    .visual-emoji { font-size: 1.5rem; }
    .visual-copy strong { display: block; font-size: 1rem; }
    .visual-copy small { display: block; color: #e3eee6; font-size: 0.78rem; }
    .vision-atlas {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.75rem 0 1.2rem;
    }
    .vision-card {
        position: relative;
        min-height: 190px;
        overflow: hidden;
        border: 1px solid rgba(23, 75, 60, 0.15);
        border-radius: 8px;
        background: #174b3c;
        isolation: isolate;
        animation: vision-arrive 520ms cubic-bezier(0.2, 0.7, 0.2, 1) both;
        animation-delay: calc(var(--tile-order, 0) * 55ms);
    }
    .vision-card img {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 450ms ease, filter 450ms ease;
    }
    .vision-card:hover img { transform: scale(1.07); filter: saturate(1.12); }
    .vision-card::after {
        position: absolute;
        inset: 0;
        z-index: 0;
        background: linear-gradient(180deg, rgba(13, 30, 25, 0.04) 8%, rgba(13, 30, 25, 0.86) 100%);
        content: "";
    }
    .vision-card-copy {
        position: absolute;
        right: 0.8rem;
        bottom: 0.75rem;
        left: 0.8rem;
        z-index: 1;
        color: #fff;
    }
    .vision-card-meta {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.35rem;
        color: #e4f1c2;
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .vision-card-emoji {
        display: inline-grid;
        width: 1.8rem;
        height: 1.8rem;
        place-items: center;
        border: 1px solid rgba(255, 255, 255, 0.55);
        border-radius: 50%;
        background: rgba(23, 75, 60, 0.76);
        font-size: 1rem;
        animation: vision-float 3.2s ease-in-out infinite;
    }
    .vision-card:nth-child(3n) .vision-card-emoji { animation-delay: -1.1s; }
    .vision-card:nth-child(3n + 1) .vision-card-emoji { animation-delay: -2s; }
    .vision-card-copy h4 {
        margin: 0 0 0.18rem;
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.94rem;
        line-height: 1.15;
    }
    .vision-card-copy p {
        margin: 0;
        color: #e5eee8;
        font-size: 0.72rem;
        line-height: 1.35;
    }
    @keyframes vision-arrive {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes vision-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { align-items: flex-start; flex-direction: column; gap: 1rem; padding: 1.2rem; }
        .hero-status { border-left: 0; border-top: 1px solid rgba(255, 255, 255, 0.3); padding: 0.7rem 0 0; }
        .stTabs [data-baseweb="tab"] { padding: 0 0.55rem; font-size: 0.82rem; }
        .visual-grid { grid-template-columns: 1fr; }
        .energy-visual { height: 132px; }
        .vision-atlas { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.55rem; }
        .vision-card { min-height: 205px; }
        .vision-card-copy { right: 0.65rem; bottom: 0.65rem; left: 0.65rem; }
    }
    @media (min-width: 701px) and (max-width: 1050px) {
        .vision-atlas { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (prefers-reduced-motion: reduce) {
        .vision-card, .vision-card-emoji { animation: none; }
        .vision-card img { transition: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
            <div>
                <div class="hero-kicker">SUSTAINABLE ENERGY INTELLIGENCE</div>
                <h1>EcoTwin AI</h1>
                <p>Observe energy use · Simulate improvements · Prioritize decarbonization</p>
            </div>
            <div class="hero-status">
                DEMO ENVIRONMENT
                <strong>SIMULATION ONLY</strong>
            </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------- Session state ----------------------------------
if "environment" not in st.session_state:
    st.session_state.environment = "Home"
if "cv_count" not in st.session_state:
    st.session_state.cv_count = 0
if "cv_active" not in st.session_state:
    st.session_state.cv_active = False
if "cv_error" not in st.session_state:
    st.session_state.cv_error = ""
if "cv_source" not in st.session_state:
    st.session_state.cv_source = "No reading yet"
if "cv_brightness" not in st.session_state:
    st.session_state.cv_brightness = 0.0
if "cv_entries" not in st.session_state:
    st.session_state.cv_entries = 0
if "cv_exits" not in st.session_state:
    st.session_state.cv_exits = 0
if "cv_heatmap" not in st.session_state:
    st.session_state.cv_heatmap = None
if "cv_updated_at" not in st.session_state:
    st.session_state.cv_updated_at = None
if "controller" not in st.session_state:
    st.session_state.controller = EcoTwinController()
if "applied_savings" not in st.session_state:
    st.session_state.applied_savings = {}
if "last_recommendation" not in st.session_state:
    st.session_state.last_recommendation = None
if "gan_lab_results" not in st.session_state:
    st.session_state.gan_lab_results = None
if "gan_anomaly_results" not in st.session_state:
    st.session_state.gan_anomaly_results = None

# --------------------------- Sidebar ----------------------------------------
with st.sidebar:
    st.header("Facility controls")
    environment = st.selectbox(
        "Choose environment",
        list(ENVIRONMENTS.keys()),
        index=list(ENVIRONMENTS.keys()).index(st.session_state.environment),
    )
    if environment != st.session_state.environment:
        st.session_state.environment = environment
        st.session_state.controller.set_environment(environment)
        st.session_state.applied_savings = {}
        st.session_state.last_recommendation = None

    st.caption(ENVIRONMENTS[environment]["description"])
    st.divider()

    st.subheader("Planning assumptions")
    period_days = st.slider("Simulation period (days)", 1, 365, 30)
    occupancy_hours = st.slider("Occupied hours per day", 0.0, 24.0, 8.0, 0.5)
    operating_hours = st.slider("Equipment operating hours per day", 1.0, 24.0, 12.0, 0.5)
    electricity_price = st.number_input(
        "Electricity price (currency/kWh)",
        min_value=0.0,
        value=DEFAULT_ELECTRICITY_PRICE_PER_KWH,
        step=0.5,
    )
    grid_factor = st.number_input(
        "Grid emissions factor (kg CO₂/kWh)",
        min_value=0.0,
        value=DEFAULT_GRID_KG_CO2_PER_KWH,
        step=0.05,
        format="%.3f",
    )
    st.caption(
        "Illustrative defaults only. Replace with your local tariff and a cited, "
        "current grid emissions factor."
    )

    st.divider()
    st.subheader("Optimization assumption")
    default_fraction = ENVIRONMENTS[environment]["savings_fraction"]
    optimization_pct = st.slider(
        "Estimated optimization potential (%)",
        min_value=0,
        max_value=40,
        value=int(default_fraction * 100),
        step=1,
    )

scenario = scenario_comparison(
    environment=environment,
    occupancy_hours_per_day=occupancy_hours,
    operating_hours_per_day=operating_hours,
    days=period_days,
    grid_kg_co2_per_kwh=grid_factor,
    electricity_price_per_kwh=electricity_price,
    savings_fraction_override=optimization_pct / 100.0,
)

# --------------------------- Shared UI helpers ------------------------------
def metric_row(data: dict) -> None:
    columns = st.columns(3)
    columns[0].metric(
        "Energy saved",
        f"{data['saved_kwh']:,.1f} kWh",
        f"{data['savings_fraction'] * 100:.1f}% estimated",
    )
    columns[1].metric(
        "Cost reduction",
        f"{data['saved_cost']:,.2f}",
        "estimated",
    )
    columns[2].metric(
        "CO₂ avoided",
        f"{data['co2_avoided_kg']:,.1f} kg",
        "estimated",
    )


def make_load_chart(data: dict) -> go.Figure:
    """Illustrative daily load profile derived from the estimated total."""
    hours = list(range(24))
    base_total = data["baseline_kwh"]
    optimized_total = data["optimized_kwh"]

    # A normalized profile to illustrate time-of-day variation, not real telemetry.
    profile = [
        0.68, 0.62, 0.58, 0.56, 0.57, 0.65,
        0.78, 0.95, 1.08, 1.12, 1.10, 1.08,
        1.05, 1.03, 1.04, 1.08, 1.15, 1.24,
        1.22, 1.12, 1.00, 0.90, 0.80, 0.72,
    ]
    total_weight = sum(profile)
    baseline = [base_total * p / total_weight / max(1, data["period_days"]) for p in profile]
    optimized = [optimized_total * p / total_weight / max(1, data["period_days"]) for p in profile]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=baseline, mode="lines+markers", name="Baseline",
        line={"width": 3, "color": "#277454"},
        marker={"size": 5},
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=optimized, mode="lines+markers", name="Optimized estimate",
        line={"width": 3, "dash": "dash", "color": "#dc8066"},
        marker={"size": 5},
    ))
    fig.update_layout(
        title="Illustrative 24-hour energy profile",
        xaxis_title="Hour of day",
        yaxis_title="Estimated energy (kWh/hour)",
        template="plotly_white",
        font={"family": "DM Sans, sans-serif", "color": "#1b2d25"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        height=370,
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#dce5dc", zeroline=False)
    fig.update_yaxes(gridcolor="#e8eee8", linecolor="#dce5dc", zeroline=False)
    return fig


# --------------------------- Main dashboard ---------------------------------
st.caption(
    "DEMO MODE · Equipment ratings, schedules, and savings assumptions are illustrative. "
    "Replace them with measured data before using the results for operational decisions."
)

overview_tab, twin_tab, cv_tab, ai_tab, devices_tab = st.tabs(
    [
        "📊 Overview",
        "🧩 Digital Twin",
        "📷 Live Computer Vision",
        "✨ GenAI Strategy",
        "⚙️ Device Controls",
    ]
)

# =========================== Overview =======================================
with overview_tab:
    st.subheader(f"Energy overview — {environment}")
    st.write(ENVIRONMENTS[environment]["description"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline energy", f"{scenario['baseline_kwh']:,.1f} kWh")
    c2.metric("Optimized estimate", f"{scenario['optimized_kwh']:,.1f} kWh")
    c3.metric("🌱 Optimization potential", f"{scenario['savings_fraction'] * 100:.1f}%")

    st.subheader("✨ Estimated savings")
    metric_row(scenario)

    st.subheader("Visual operations")
    vision_source = st.session_state.cv_source
    vision_has_reading = vision_source != "No reading yet"
    vision_capacity = int(st.session_state.get("cv_room_capacity", 30))
    vision_count = int(st.session_state.cv_count)
    vision_live = vision_source == "Live camera"
    vision_metrics = st.columns(4)
    vision_metrics[0].metric("Vision source", vision_source)
    vision_metrics[1].metric(
        "Occupancy proxy",
        f"{vision_count} / {vision_capacity}" if vision_has_reading else "Awaiting reading",
        f"{min(100, vision_count / vision_capacity * 100):.0f}% of capacity" if vision_has_reading else None,
    )
    vision_metrics[2].metric(
        "Entry / exit crossings",
        f"{st.session_state.cv_entries} / {st.session_state.cv_exits}" if vision_live else "Live camera only",
    )
    vision_metrics[3].metric(
        "Relative brightness",
        f"{st.session_state.cv_brightness:.0f} / 255" if vision_live else "Not available",
        "Scene cue, not a light-meter reading" if vision_live else None,
    )

    heatmap_col, capability_col = st.columns([1.2, 1])
    with heatmap_col:
        st.markdown("#### Recent activity map")
        if vision_live and st.session_state.cv_heatmap is not None:
            overview_heatmap = go.Figure(go.Heatmap(
                z=np.log1p(np.asarray(st.session_state.cv_heatmap, dtype=float)),
                colorscale=[[0, "#2d78a5"], [0.5, "#e7cc70"], [1, "#d85c43"]],
                colorbar={"title": "Activity"},
            ))
            overview_heatmap.update_layout(
                height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                xaxis={"visible": False}, yaxis={"visible": False, "autorange": "reversed"},
            )
            st.plotly_chart(overview_heatmap, use_container_width=True, key="overview_cv_heatmap")
            st.caption("Accumulated person-center detections in the camera view; not a floor plan or unique visitor count.")
        else:
            st.info("Start the webcam in Live Computer Vision to populate the activity map.")
    with capability_col:
        st.markdown("#### Vision-to-action coverage")
        st.markdown(
            "**Occupancy and flow**  · People count, capacity, entry/exit line, recent trend  \n"
            "**Energy cues**  · Empty-room checks, operator-confirmed lights/HVAC, daylight scene cue  \n"
            "**Space and maintenance**  · Activity heatmap, zone twin, solar/HVAC/opening review  \n"
            "**Energy advisor**  · Visual context paired with tariff and grid-factor estimates"
        )
        st.caption("Camera observations are not electricity measurements. Confirm maintenance findings and validate savings with meter or device telemetry.")

    st.markdown(
        """
        <div class="visual-grid">
          <article class="energy-visual">
            <img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=900&q=80" alt="Solar panels collecting sunlight">
            <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">☀️</span><div><strong>Solar generation</strong><small>Clean power from daylight</small></div></div>
          </article>
          <article class="energy-visual">
            <img src="https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=900&q=80" alt="Wind turbines generating renewable electricity">
            <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">🌬️</span><div><strong>Wind generation</strong><small>Renewable power at scale</small></div></div>
          </article>
          <article class="energy-visual">
            <img src="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=900&q=80" alt="Modern high-rise building in daylight">
            <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">🏙️</span><div><strong>Efficient buildings</strong><small>Smarter controls, lower demand</small></div></div>
          </article>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.5, 1])
    with left:
        st.plotly_chart(make_load_chart(scenario), use_container_width=True)
    with right:
        st.markdown("#### Facility snapshot")
        st.write(f"**Facility:** {environment}")
        st.write(f"**Period:** {period_days} day(s)")
        st.write(f"**Occupied:** {occupancy_hours:g} hours/day")
        st.write(f"**Operating:** {operating_hours:g} hours/day")
        st.write(f"**Tariff:** {electricity_price:.2f} per kWh")
        st.write(f"**Grid factor:** {grid_factor:.3f} kg CO₂/kWh")
        st.info(
            "The chart uses an illustrative load shape. It is not a live meter trace "
            "or a prediction of exact hourly demand."
        )

    st.subheader("Prioritized strategy ideas")
    for i, strategy in enumerate(ENVIRONMENTS[environment]["strategies"], start=1):
        st.markdown(f"{i}. {strategy}")

    st.subheader("Baseline vs. optimized")
    compare_df = pd.DataFrame(
        [
            {
                "Metric": "Energy (kWh)",
                "Baseline": scenario["baseline_kwh"],
                "Optimized estimate": scenario["optimized_kwh"],
                "Estimated reduction": scenario["saved_kwh"],
            },
            {
                "Metric": "Electricity cost",
                "Baseline": scenario["baseline_cost"],
                "Optimized estimate": scenario["optimized_cost"],
                "Estimated reduction": scenario["saved_cost"],
            },
            {
                "Metric": "CO₂ (kg)",
                "Baseline": scenario["baseline_co2_kg"],
                "Optimized estimate": scenario["optimized_co2_kg"],
                "Estimated reduction": scenario["co2_avoided_kg"],
            },
        ]
    )
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download scenario results (CSV)",
        data=compare_df.to_csv(index=False).encode("utf-8"),
        file_name="ecotwin_scenario_results.csv",
        mime="text/csv",
    )

# =========================== Digital Twin ===================================
with twin_tab:
    st.subheader("Digital Twin simulation")
    st.write(
        "Use the controls below to explore equipment status and occupancy-triggered "
        "lighting. The controller is a local simulation, not a building-management system."
    )

    controller: EcoTwinController = st.session_state.controller
    controller.electricity_price_per_kwh = electricity_price
    controller.grid_kg_co2_per_kwh = grid_factor

    occ_col, delay_col = st.columns(2)
    with occ_col:
        simulated_occupancy = st.number_input(
            "Simulated people in facility",
            min_value=0,
            max_value=1000,
            value=int(controller.occupancy),
            step=1,
            key="simulated_occupancy",
        )
    with delay_col:
        vacancy_delay = st.number_input(
            "Vacancy delay before auto-off (seconds)",
            min_value=1,
            max_value=3600,
            value=int(controller.vacancy_delay_seconds),
            step=5,
        )

    controller.vacancy_delay_seconds = int(vacancy_delay)
    if st.button("Update simulated occupancy", key="update_occ"):
        controller.set_occupancy(int(simulated_occupancy))
        st.success("Simulation occupancy updated.")

    st.markdown("#### Equipment snapshot")
    status = pd.DataFrame(controller.get_device_status())
    st.dataframe(status, use_container_width=True, hide_index=True)

    power_col, energy_col, savings_col = st.columns(3)
    power_col.metric("Current simulated power", f"{controller.current_power_watts():,.0f} W")
    energy_col.metric("Simulated energy", f"{controller.energy_used_kwh:.5f} kWh")
    savings_col.metric("Estimated auto-off savings", f"{controller.energy_saved_kwh:.5f} kWh")

    step_col, reset_col = st.columns(2)
    with step_col:
        seconds = st.number_input(
            "Advance simulation by seconds",
            min_value=1,
            max_value=3600,
            value=60,
            step=1,
        )
        if st.button("Advance simulation", key="advance_sim"):
            controller.tick(seconds)
            st.rerun()
    with reset_col:
        st.write("")
        st.write("")
        if st.button("Reset simulation", key="reset_sim"):
            controller.reset()
            st.rerun()

    if controller.events:
        st.markdown("#### Recent events")
        st.dataframe(
            pd.DataFrame({"Event": controller.events}),
            use_container_width=True,
            hide_index=True,
        )

# =========================== Computer Vision ================================
with cv_tab:
    st.subheader("Live visual digital twin")
    st.caption(
        "Camera observations are approximate. Visual detections do not measure electricity; "
        "energy values below use rated device power and are illustrative unless backed by meters."
    )
    st.warning(
        "Person detection can miss or miscount people. This prototype does not identify people, "
        "and must not be used for safety, access control, or automatic equipment control."
    )
    st.markdown(
        """
        <div class="vision-atlas" aria-label="EcoTwin computer vision capabilities">
          <article class="vision-card" style="--tile-order:0"><img src="https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?auto=compress&cs=tinysrgb&w=900" alt="People collaborating in a bright shared workspace"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">👥</span>01 · SPACE</div><h4>Smart occupancy</h4><p>Counts visible people and estimates room utilization.</p></div></article>
          <article class="vision-card" style="--tile-order:1"><img src="https://images.unsplash.com/photo-1497366754035-f200968a6e72?auto=format&fit=crop&w=900&q=85" alt="Quiet, unoccupied modern office ready for an energy check"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">💡</span>02 · WASTE</div><h4>Empty-room energy</h4><p>Flags reported-on lights or HVAC when no people are detected.</p></div></article>
          <article class="vision-card" style="--tile-order:2"><img src="https://images.unsplash.com/photo-1497366811353-6870744d04b2?auto=format&fit=crop&w=900&q=85" alt="Daylight filling a contemporary office near its windows"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">☀️</span>03 · DAYLIGHT</div><h4>Daylight detection</h4><p>Uses relative frame brightness to prompt a light-sensor check.</p></div></article>
          <article class="vision-card" style="--tile-order:3"><img src="https://images.pexels.com/photos/1181406/pexels-photo-1181406.jpeg?auto=compress&cs=tinysrgb&w=900" alt="Open collaborative workspace showing activity zones"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🗺️</span>04 · ACTIVITY</div><h4>Occupancy heatmap</h4><p>Maps repeated person detections into relative camera-view activity.</p></div></article>
          <article class="vision-card" style="--tile-order:4"><img src="https://images.unsplash.com/photo-1621905251189-08b45d6a269e?auto=format&fit=crop&w=900&q=85" alt="Technician inspecting building equipment"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🛠️</span>05 · MAINTENANCE</div><h4>HVAC inspection</h4><p>Collects operator-confirmed vent, filter, and equipment concerns.</p></div></article>
          <article class="vision-card" style="--tile-order:5"><img src="https://images.unsplash.com/photo-1524758631624-e2822e304c36?auto=format&fit=crop&w=900&q=85" alt="People moving through a shared office environment"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🚶</span>06 · FLOW</div><h4>People flow</h4><p>Tracks anonymous person IDs across a virtual entry/exit line.</p></div></article>
          <article class="vision-card" style="--tile-order:6"><img src="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=900&q=85" alt="Modern building facade representing a digital twin zone view"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🏙️</span>07 · DIGITAL TWIN</div><h4>Live visual twin</h4><p>Shows camera zones beside people counts and allocated rated load.</p></div></article>
          <article class="vision-card" style="--tile-order:7"><img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=900&q=85" alt="Solar panels under bright daylight for renewable inspection"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🔆</span>08 · SOLAR</div><h4>Solar inspection</h4><p>Lets staff record visible panel dirt, debris, or damage for review.</p></div></article>
          <article class="vision-card" style="--tile-order:8"><img src="https://images.pexels.com/photos/439227/pexels-photo-439227.jpeg?auto=compress&cs=tinysrgb&w=900" alt="Window and door details for an opening-status inspection"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🪟</span>09 · OPENINGS</div><h4>Window / door check</h4><p>Pairs operator-reported open status with HVAC operation.</p></div></article>
          <article class="vision-card" style="--tile-order:9"><img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=85" alt="Energy analytics dashboard supporting a facility recommendation"><div class="vision-card-copy"><div class="vision-card-meta"><span class="vision-card-emoji">🧠</span>10 · ADVISOR</div><h4>Energy AI advisor</h4><p>Turns visual cues and tariff assumptions into reviewable actions.</p></div></article>
        </div>
        """,
        unsafe_allow_html=True,
    )

    live_tab, inspection_tab = st.tabs(["Live room", "Inspections & advisor"])

    try:
        import av
        import cv2
        from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
        from ultralytics import YOLO

        class PersonDetectionProcessor(VideoProcessorBase):
            def __init__(self):
                self.model = YOLO("yolov8n.pt")
                self.count = 0
                self.centers: list[tuple[float, float]] = []
                self.heatmap = np.zeros((8, 12), dtype=float)
                self.entries = 0
                self.exits = 0
                self.tracks: dict[int, tuple[float, float, float]] = {}
                self.crossed_at: dict[int, float] = {}
                self.brightness = 0.0
                self.samples: list[tuple[float, int]] = []
                self.last_sample = 0.0
                self.last_error = ""
                self.lock = threading.Lock()

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")
                try:
                    result = self.model.track(
                        image,
                        persist=True,
                        classes=[0],
                        conf=0.35,
                        tracker="bytetrack.yaml",
                        verbose=False,
                    )[0]
                    height, width = image.shape[:2]
                    centers: list[tuple[float, float]] = []
                    entries = 0
                    exits = 0
                    tracked_boxes = result.boxes
                    if tracked_boxes is not None and len(tracked_boxes):
                        xyxy = tracked_boxes.xyxy.cpu().numpy()
                        track_ids = (
                            tracked_boxes.id.int().cpu().tolist()
                            if tracked_boxes.id is not None
                            else list(range(len(xyxy)))
                        )
                    now = time.monotonic()
                    with self.lock:
                        for box, track_id in zip(xyxy if tracked_boxes is not None and len(tracked_boxes) else [], track_ids if tracked_boxes is not None and len(tracked_boxes) else []):
                            x = float((box[0] + box[2]) / 2 / width)
                            y = float((box[1] + box[3]) / 2 / height)
                            centers.append((x, y))
                            track_id = int(track_id)
                            previous = self.tracks.get(track_id)
                            if previous and now - self.crossed_at.get(track_id, 0.0) > 2:
                                if previous[1] < 0.5 <= y:
                                    entries += 1
                                    self.crossed_at[track_id] = now
                                elif previous[1] >= 0.5 > y:
                                    exits += 1
                                    self.crossed_at[track_id] = now
                            self.tracks[track_id] = (x, y, now)

                        self.count = len(centers)
                        self.centers = centers
                        self.entries += entries
                        self.exits += exits
                        self.heatmap *= 0.995
                        for x, y in centers:
                            row = min(7, int(y * 8))
                            column = min(11, int(x * 12))
                            self.heatmap[row, column] += 1
                        self.brightness = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
                        if now - self.last_sample >= 1:
                            self.samples.append((now, self.count))
                            self.samples = self.samples[-3600:]
                            self.last_sample = now
                        self.tracks = {
                            key: value for key, value in self.tracks.items()
                            if now - value[2] < 10
                        }
                        self.crossed_at = {
                            key: value for key, value in self.crossed_at.items()
                            if now - value < 10
                        }

                    annotated = result.plot()
                    cv2.line(annotated, (0, height // 2), (width, height // 2), (30, 200, 230), 2)
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception as exc:
                    with self.lock:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                    return av.VideoFrame.from_ndarray(image, format="bgr24")

            def get_snapshot(self):
                with self.lock:
                    return {
                        "count": self.count,
                        "centers": list(self.centers),
                        "heatmap": self.heatmap.copy(),
                        "entries": self.entries,
                        "exits": self.exits,
                        "brightness": self.brightness,
                        "samples": list(self.samples),
                        "last_error": self.last_error,
                    }

        with live_tab:
            config_col, equipment_col = st.columns([1, 1])
            with config_col:
                room_capacity = st.number_input("Room capacity", min_value=1, max_value=1000, value=30, key="cv_room_capacity")
                crowd_threshold = st.slider("Crowding alert threshold (%)", 10, 100, 80, 5)
                vacancy_minutes = st.slider("Empty-room alert delay (minutes)", 1, 60, 20)
            with equipment_col:
                st.markdown("#### Operator-confirmed equipment state")
                lighting_devices = [d for d in st.session_state.controller.available_devices if d["kind"] == "lighting"]
                hvac_devices = [d for d in st.session_state.controller.available_devices if d["kind"] == "hvac"]
                lights_default = any(st.session_state.controller.devices_on.get(d["name"], False) for d in lighting_devices)
                hvac_default = any(st.session_state.controller.devices_on.get(d["name"], False) for d in hvac_devices)
                lights_on = st.toggle("Lights are on", value=lights_default, key="cv_lights_on")
                hvac_on = st.toggle("HVAC is running", value=hvac_default, key="cv_hvac_on")
                st.caption("These are operator inputs, not detections from the camera.")

            ctx = webrtc_streamer(
                key="ecotwin-person-detection",
                video_processor_factory=PersonDetectionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.video_processor:
                processor = ctx.video_processor

                @st.fragment(run_every="2s")
                def render_live_insights():
                    snapshot = processor.get_snapshot()
                    detected_count = snapshot["count"]
                    st.session_state.cv_count = detected_count
                    st.session_state.cv_source = "Live camera"
                    st.session_state.cv_entries = snapshot["entries"]
                    st.session_state.cv_exits = snapshot["exits"]
                    st.session_state.cv_heatmap = snapshot["heatmap"].tolist()
                    utilization = min(100.0, detected_count / room_capacity * 100)
                    metric_cols = st.columns(4)
                    metric_cols[0].metric("People detected", detected_count)
                    metric_cols[1].metric("Estimated occupancy", f"{utilization:.0f}%")
                    metric_cols[2].metric("Flow across virtual line", f"{snapshot['entries']} in / {snapshot['exits']} out")
                    metric_cols[3].metric("Relative frame brightness", f"{snapshot['brightness']:.0f} / 255")
                    st.caption("Virtual line direction: top-to-bottom counts as entry; reverse the camera orientation if needed.")

                    now = time.monotonic()
                    st.session_state.cv_brightness = snapshot["brightness"]
                    st.session_state.cv_updated_at = datetime.now()
                    vacant_since = st.session_state.get("cv_vacant_since")
                    if detected_count == 0:
                        if vacant_since is None:
                            vacant_since = now
                            st.session_state.cv_vacant_since = vacant_since
                    else:
                        vacant_since = None
                        st.session_state.cv_vacant_since = None
                    vacant_seconds = max(0.0, now - vacant_since) if vacant_since else 0.0
                    controller = st.session_state.controller
                    st.button("Use detected count in Digital Twin", key="apply_live_occupancy", on_click=controller.set_occupancy, args=(detected_count,))

                    if utilization >= crowd_threshold:
                        st.warning("Crowding threshold reached. Check the space and its operating plan.")
                    waste_devices = [
                        d for d in controller.available_devices
                        if (d["kind"] == "lighting" and lights_on) or (d["kind"] == "hvac" and hvac_on)
                    ]
                    if detected_count == 0 and (lights_on or hvac_on):
                        st.warning(
                            f"Potential empty-room energy waste: equipment is reported on during "
                            f"{vacant_seconds / 60:.1f} vacant minute(s). Verify before acting."
                        )
                        if vacant_seconds >= vacancy_minutes * 60:
                            rated_watts = sum(float(d["watts"]) for d in waste_devices)
                            estimated_kwh = rated_watts * vacant_seconds / 3_600_000
                            st.info(
                                f"At listed rated power, the illustrative avoidable load is "
                                f"{estimated_kwh:.4f} kWh, {estimated_kwh * electricity_price:.2f} cost units, "
                                f"and {estimated_kwh * grid_factor:.4f} kg CO₂. This is not meter-verified."
                            )

                    daylight_col, twin_col = st.columns([1.3, 1])
                    with daylight_col:
                        st.markdown("#### Occupancy heatmap")
                        heat = np.log1p(snapshot["heatmap"])
                        heat_fig = go.Figure(go.Heatmap(
                            z=heat,
                            colorscale=[[0, "#2d78a5"], [0.5, "#e7cc70"], [1, "#d85c43"]],
                            colorbar={"title": "Relative activity"},
                        ))
                        heat_fig.update_layout(
                            height=300, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                            xaxis={"visible": False}, yaxis={"visible": False, "autorange": "reversed"},
                        )
                        st.plotly_chart(heat_fig, use_container_width=True, key="cv_heatmap")
                        st.caption("Color shows accumulated person-center detections in this camera view, not unique visitors or a floor plan.")
                        brightness = snapshot["brightness"]
                        if brightness >= 155:
                            st.success("Bright scene: check whether daylight can support dimmer artificial lighting.")
                        else:
                            st.info("No strong bright-scene cue. Camera brightness is not a calibrated daylight measurement.")
                    with twin_col:
                        st.markdown("#### Live visual digital twin")
                        zone_counts = [0, 0, 0, 0]
                        for x, y in snapshot["centers"]:
                            zone = (2 if y >= 0.5 else 0) + (1 if x >= 0.5 else 0)
                            zone_counts[zone] += 1
                        total_watts = controller.current_power_watts()
                        zone_names = ["Northwest", "Northeast", "Southwest", "Southeast"]
                        zone_df = pd.DataFrame({
                            "Camera zone": zone_names,
                            "People": zone_counts,
                            "Allocated rated load (W)": [round(total_watts / 4, 1)] * 4,
                        })
                        st.dataframe(zone_df, use_container_width=True, hide_index=True)
                        st.caption("Rated load is evenly allocated for visualization; zone power is not sensed.")

                    samples = snapshot["samples"]
                    if len(samples) > 1:
                        flow_fig = go.Figure(go.Scatter(
                            x=[datetime.fromtimestamp(time.time() - (now - stamp)) for stamp, _ in samples],
                            y=[count for _, count in samples], mode="lines", name="People detected",
                            line={"color": "#277454", "width": 2},
                        ))
                        flow_fig.update_layout(
                            title="Recent occupancy trend", xaxis_title="Time", yaxis_title="People",
                            height=250, template="plotly_white", margin={"l": 20, "r": 20, "t": 45, "b": 25},
                        )
                        st.plotly_chart(flow_fig, use_container_width=True, key="cv_flow_history")
                    if snapshot["last_error"]:
                        st.error(snapshot["last_error"])

                render_live_insights()
            else:
                st.info("Allow camera access and start the webcam stream to begin live observations.")
                demo_count = st.slider("Demo people count", 0, 50, int(st.session_state.cv_count), key="cv_demo_count")
                st.session_state.cv_count = demo_count
                st.session_state.cv_source = "Demo input"
                st.session_state.cv_heatmap = None
                if st.button("Use demo occupancy in Digital Twin", key="use_demo_occupancy"):
                    st.session_state.controller.set_occupancy(demo_count)
                    st.success("Demo occupancy copied to the Digital Twin simulation.")

        with inspection_tab:
            st.markdown("#### Visual evidence review")
            st.caption("Upload a room or equipment image. The prototype records operator findings; it has no trained vent, window, door, or solar-defect classifier.")
            evidence = st.file_uploader("Inspection image", type=["jpg", "jpeg", "png"], key="cv_inspection_image")
            if evidence:
                image_bytes = np.frombuffer(evidence.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded inspection evidence", use_container_width=True)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    st.caption(
                        f"Image screening only: mean brightness {gray.mean():.1f}/255; "
                        f"sharpness cue {cv2.Laplacian(gray, cv2.CV_64F).var():.1f}. "
                        "These values do not diagnose dirt, damage, or daylight."
                    )

            solar_col, equipment_col, opening_col = st.columns(3)
            with solar_col:
                st.markdown("#### Solar panel inspection")
                solar_findings = st.multiselect(
                    "Visible solar-panel concerns (operator confirmed)",
                    ["Dirt or debris", "Crack or surface damage", "Possible shading", "No visible concern"],
                    key="cv_solar_findings",
                )
                if solar_findings and "No visible concern" not in solar_findings:
                    st.warning("Solar maintenance review recommended; confirm on-site before scheduling work.")
                else:
                    st.info("No operator-confirmed solar concern recorded.")
            with equipment_col:
                st.markdown("#### HVAC / equipment inspection")
                equipment_findings = st.multiselect(
                    "Visible concerns (operator confirmed)",
                    ["Blocked air vent", "Obstructed radiator", "Dirty filter evidence", "Visible equipment damage"],
                    key="cv_equipment_findings",
                )
                if equipment_findings:
                    st.warning("Maintenance alert for human review: " + ", ".join(equipment_findings))
                else:
                    st.info("No operator-confirmed equipment concern recorded.")
            with opening_col:
                st.markdown("#### Window / door + HVAC")
                opening_status = st.selectbox("Window or door status", ["Unknown", "Closed", "Open"], key="cv_opening_status")
                hvac_running = st.toggle("HVAC reported running", key="cv_opening_hvac")
                if opening_status == "Open" and hvac_running:
                    st.warning("Potential conditioned-air loss. Verify the opening and review HVAC settings.")
                elif opening_status == "Unknown":
                    st.caption("Operator confirmation is needed; open/closed state is not detected.")

            st.markdown("#### Vision-powered energy AI advisor")
            advisor_items = []
            if st.session_state.cv_count == 0 and (lights_on or hvac_on):
                advisor_items.append("Confirm the room is empty, then review lighting and cooling schedules.")
            if st.session_state.get("cv_brightness", 0) >= 155:
                advisor_items.append("Check light-sensor readings before dimming lights in the bright camera view.")
            if opening_status == "Open" and hvac_running:
                advisor_items.append("Check the open window/door and HVAC operating plan to limit conditioned-air loss.")
            if equipment_findings or solar_findings:
                advisor_items.append("Route the operator-confirmed visual concern to facilities maintenance for inspection.")
            if not advisor_items:
                advisor_items.append("No current visual action is indicated; continue validating with occupancy and energy telemetry.")
            for item in advisor_items:
                st.markdown(f"- {item}")
            st.write(
                f"Planning context for {period_days} day(s): {scenario['saved_kwh']:,.1f} kWh, "
                f"{scenario['saved_cost']:,.2f} cost units, and "
                f"{scenario['co2_avoided_kg']:,.1f} kg CO₂ are estimated for the selected "
                "Digital Twin scenario using the configured tariff and grid factor."
            )
            st.caption("Recommendations are prompts for human review. Savings require smart-meter or device-telemetry data; simulation figures are labeled estimates.")

    except Exception as exc:
        with live_tab:
            st.info(
                "Live detection is unavailable in this environment. Install the computer-vision "
                "dependencies and restart the app to enable webcam features."
            )
            st.caption(f"Live CV detail: {type(exc).__name__}: {exc}")
            demo_count = st.slider("Demo people count", 0, 50, int(st.session_state.cv_count), key="cv_fallback_count")
            st.session_state.cv_count = demo_count
            st.session_state.cv_source = "Demo input"
            st.session_state.cv_heatmap = None
            if st.button("Use demo occupancy in Digital Twin", key="use_fallback_occupancy"):
                st.session_state.controller.set_occupancy(demo_count)
                st.success("Demo occupancy copied to the Digital Twin simulation.")

# =========================== GenAI Strategy ================================
with ai_tab:
    gan_tab, advisor_tab = st.tabs(["EcoTwin GAN Lab", "AI Strategy Advisor"])

    with gan_tab:
        st.subheader("Generate · Simulate · Optimize")
        st.markdown(
            """
            <div class="visual-grid">
              <article class="energy-visual">
                <img src="https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=900&q=85" alt="Energy data visualized in an analytics dashboard">
                <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">▥</span><div><strong>Conditional profiles</strong><small>Occupancy · weather · equipment</small></div></div>
              </article>
              <article class="energy-visual">
                <img src="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=900&q=85" alt="Building facade representing operational strategy scenarios">
                <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">⌁</span><div><strong>Strategy discovery</strong><small>Evaluate comfort and energy together</small></div></div>
              </article>
              <article class="energy-visual">
                <img src="https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=900&q=85" alt="Solar panels representing renewable-aware energy scenarios">
                <div class="visual-copy"><span class="visual-emoji" aria-hidden="true">☀</span><div><strong>Carbon-aware operation</strong><small>Renewables · tariff · grid factor</small></div></div>
              </article>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(
            "Prototype generator: conditional statistical profiles, not a trained GAN. "
            "Upload meter history to anchor the baseline; train and validate a CGAN/TimeGAN "
            "on held-out facility data before describing outputs as learned GAN results."
        )

        meter_file = st.file_uploader(
            "Optional hourly meter history (CSV)",
            type=["csv"],
            key="gan_meter_csv",
            help="Use timestamp and energy columns. Occupancy is optional. At least 30 readings are needed for anomaly scoring.",
        )
        meter_data = None
        energy_column = None
        timestamp_column = None
        occupancy_column = None
        historical_hourly = None
        if meter_file is not None:
            try:
                meter_data = pd.read_csv(meter_file.getvalue())
                numeric_columns = meter_data.select_dtypes(include="number").columns.tolist()
                if not numeric_columns:
                    st.warning("The CSV needs a numeric energy column.")
                    meter_data = None
                else:
                    preferred_energy = next(
                        (name for name in numeric_columns if name.lower() in {"energy_kwh", "kwh", "consumption_kwh"}),
                        numeric_columns[0],
                    )
                    energy_column = st.selectbox(
                        "Energy column (kWh)", numeric_columns,
                        index=numeric_columns.index(preferred_energy), key="gan_energy_column",
                    )
                    timestamp_options = ["No timestamp"] + meter_data.columns.tolist()
                    preferred_timestamp = next(
                        (name for name in meter_data.columns if "time" in name.lower() or "date" in name.lower()),
                        "No timestamp",
                    )
                    timestamp_column = st.selectbox(
                        "Timestamp column", timestamp_options,
                        index=timestamp_options.index(preferred_timestamp), key="gan_timestamp_column",
                    )
                    occupancy_options = ["None"] + numeric_columns
                    occupancy_column = st.selectbox("Optional occupancy column", occupancy_options, key="gan_occupancy_column")
                    if timestamp_column != "No timestamp":
                        parsed_times = pd.to_datetime(meter_data[timestamp_column], errors="coerce")
                        hourly_history = (
                            pd.DataFrame({"hour": parsed_times.dt.hour, "energy": pd.to_numeric(meter_data[energy_column], errors="coerce")})
                            .dropna()
                            .groupby("hour")["energy"]
                            .mean()
                            .reindex(range(24))
                        )
                        if hourly_history.notna().sum() >= 8:
                            historical_hourly = hourly_history.interpolate(limit_direction="both").ffill().bfill().tolist()
                            st.success("Meter history will anchor the 24-hour baseline profile.")
                        else:
                            st.warning("Need readings across at least 8 different hours to form an hourly baseline; generation will use equipment assumptions.")
            except Exception as exc:
                st.error(f"Could not read the CSV: {type(exc).__name__}: {exc}")
                meter_data = None

        st.markdown("#### Scenario conditions")
        condition_cols = st.columns(4)
        with condition_cols[0]:
            profile_count = st.slider("Profiles to generate", 50, 5000, 500, 50, key="gan_profile_count")
        with condition_cols[1]:
            outdoor_temp = st.number_input("Outdoor temperature (°C)", -10.0, 50.0, 28.0, 1.0, key="gan_outdoor_temp")
        with condition_cols[2]:
            solar_pct = st.slider("Solar availability (%)", 0, 100, 35, 5, key="gan_solar_pct")
        with condition_cols[3]:
            comfort_limit = st.slider("Comfort-change limit (°C)", 0.0, 3.0, 1.5, 0.1, key="gan_comfort_limit")

        vision_available = st.session_state.cv_source in {"Live camera", "Demo input"}
        use_vision_condition = st.checkbox(
            "Include the latest computer-vision occupancy condition",
            value=vision_available,
            disabled=not vision_available,
            key="gan_use_vision_condition",
            help="The camera count is a snapshot condition, not an energy meter or a full-day occupancy schedule.",
        )
        condition_occupancy = int(st.session_state.cv_count) if use_vision_condition else 8
        occupancy_label = f"{condition_occupancy} people from {st.session_state.cv_source}" if use_vision_condition else "Default profile condition"
        st.caption(
            f"Occupancy condition: {occupancy_label}. The sidebar's occupied-hours assumption sets the generated daily schedule. "
            "Tariffs use an illustrative 0.8× overnight / 1.25× evening peak profile; edit the base tariff in Facility controls."
        )

        if st.button("Generate and rank scenarios", type="primary", key="run_gan_lab"):
            with st.spinner(f"Evaluating {profile_count:,} conditional profiles..."):
                try:
                    st.session_state.gan_lab_results = generate_synthetic_energy_scenarios(
                        environment=environment,
                        profile_count=profile_count,
                        operating_hours_per_day=operating_hours,
                        occupancy_hours_per_day=occupancy_hours,
                        current_occupancy=condition_occupancy,
                        outdoor_temperature_c=outdoor_temp,
                        solar_availability=solar_pct / 100,
                        electricity_price_per_kwh=electricity_price,
                        grid_kg_co2_per_kwh=grid_factor,
                        comfort_limit_c=comfort_limit,
                        historical_hourly_kwh=historical_hourly,
                    )
                except Exception as exc:
                    st.error(f"Scenario generation failed: {type(exc).__name__}: {exc}")

        lab_results = st.session_state.gan_lab_results
        if lab_results:
            ranking = lab_results["ranking"]
            baseline_row = ranking.loc[ranking["Strategy"] == "Baseline schedule"].iloc[0]
            alternatives = ranking[(ranking["Strategy"] != "Baseline schedule") & ranking["Feasible"]]
            best_row = alternatives.iloc[0] if not alternatives.empty else baseline_row
            metric_cols = st.columns(4)
            metric_cols[0].metric("Generated profiles", f"{profile_count:,}", "conditional scenarios")
            metric_cols[1].metric("Baseline energy", f"{baseline_row['Daily energy (kWh)']:.2f} kWh/day")
            metric_cols[2].metric("Top feasible strategy", best_row["Strategy"])
            metric_cols[3].metric("Estimated grid CO₂", f"{best_row['Grid emissions (kg CO2)']:.2f} kg/day")

            detail_cols = st.columns(3)
            detail_cols[0].metric("Proposed energy", f"{best_row['Daily energy (kWh)']:.2f} kWh/day", f"{best_row['Energy saved (%)']:.1f}% vs baseline")
            detail_cols[1].metric("Estimated daily cost", f"{best_row['Daily cost']:.2f}", f"{best_row['Cost saved (%)']:.1f}% vs baseline")
            detail_cols[2].metric("Comfort change", f"{best_row['Comfort change (°C)']:.1f} °C", "within limit" if best_row["Feasible"] else "not feasible")

            chart_col, ranking_col = st.columns([1.15, 1])
            with chart_col:
                st.markdown("#### Baseline vs leading strategy")
                chart_data = lab_results["hourly"]
                chart = go.Figure()
                for label, color, dash in [
                    ("Baseline schedule", "#277454", "solid"),
                    (best_row["Strategy"], "#dc8066", "dash"),
                ]:
                    series = chart_data[chart_data["Strategy"] == label]
                    chart.add_trace(go.Scatter(
                        x=series["Hour"], y=series["Energy (kWh)"], name=label,
                        mode="lines+markers", line={"color": color, "dash": dash, "width": 3},
                    ))
                chart.update_layout(
                    height=330, template="plotly_white", xaxis_title="Hour of day",
                    yaxis_title="Generated mean energy (kWh)", margin={"l": 20, "r": 20, "t": 30, "b": 30},
                )
                st.plotly_chart(chart, use_container_width=True, key="gan_scenario_chart")
            with ranking_col:
                st.markdown("#### Evaluator ranking")
                st.dataframe(ranking, use_container_width=True, hide_index=True)
                st.caption(
                    "Objective weights: energy 40%, cost 30%, grid emissions 25%, comfort 5%. "
                    "Comfort-limit violations are marked infeasible and excluded from the recommended candidate. "
                    "Tariff bands and solar offset are explicit simulation assumptions."
                )

            st.download_button(
                "Download generated hourly profiles (CSV)",
                data=lab_results["synthetic"].to_csv(index=False).encode("utf-8"),
                file_name="ecotwin_synthetic_energy_profiles.csv",
                mime="text/csv",
                key="download_gan_profiles",
            )
            st.caption("The generator proposes profiles; a separate evaluator ranks feasible strategies. Savings, cost, and emissions are estimates, not guaranteed outcomes.")

        st.markdown("#### Meter anomaly review")
        if meter_data is not None and energy_column is not None:
            if st.button("Score uploaded meter readings", key="score_meter_anomalies"):
                try:
                    scored = detect_energy_anomalies(
                        meter_data,
                        energy_column,
                        timestamp_column if timestamp_column != "No timestamp" else None,
                        occupancy_column if occupancy_column != "None" else None,
                    )
                    st.session_state.gan_anomaly_results = scored
                except Exception as exc:
                    st.error(f"Anomaly review unavailable: {exc}")
            anomaly_results = st.session_state.gan_anomaly_results
            if anomaly_results is not None:
                anomalies = anomaly_results[anomaly_results["Anomaly"]].sort_values("Anomaly score")
                if anomalies.empty:
                    st.success("No unusual rows were flagged in the uploaded sample.")
                else:
                    st.warning(f"{len(anomalies)} unusual meter row(s) flagged for human review; an anomaly is not a confirmed fault.")
                    st.dataframe(anomalies.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Upload meter history above to score unusual consumption patterns. Without meter readings, anomaly detection is unavailable rather than simulated as a real alert.")

    with advisor_tab:
        st.subheader("EcoTwin AI strategy advisor")
        st.write(
            "The local evaluator computes scenario metrics. The language model can explain and prioritize actions, "
            "but it does not calculate or verify energy use."
        )
        if st.session_state.gan_lab_results:
            ranked = st.session_state.gan_lab_results["ranking"]
            best = ranked[(ranked["Strategy"] != "Baseline schedule") & ranked["Feasible"]]
            if not best.empty:
                st.info(f"Leading feasible lab candidate: {best.iloc[0]['Strategy']} · {best.iloc[0]['Energy saved (%)']:.1f}% estimated energy change under current assumptions.")

        api_key_present = bool(os.getenv("OPENAI_API_KEY"))
        if api_key_present:
            st.success("OPENAI_API_KEY is configured in this process.")
        else:
            st.info("OPENAI_API_KEY is not configured. The advisor will use the built-in rule-based fallback.")

        model_name = st.text_input("OpenAI model", value="gpt-4o-mini", key="advisor_model")
        vacancy_detected = st.checkbox(
            "Current facility is vacant",
            value=(st.session_state.cv_count == 0),
            help="This is an optional contextual hint; it does not override facility safety.",
            key="advisor_vacant",
        )

        if st.button("Generate recommendations", type="primary", key="generate_advice"):
            with st.spinner("Preparing recommendations..."):
                result = generate_ai_recommendations(
                    scenario=scenario,
                    occupancy_count=int(st.session_state.cv_count),
                    vacancy_detected=vacancy_detected,
                    model=model_name.strip() or "gpt-4o-mini",
                )
                st.session_state.last_recommendation = result

        recommendation = st.session_state.last_recommendation
        if recommendation:
            if recommendation.get("fallback"):
                st.warning(f"Using {recommendation['provider']}.")
                st.caption(recommendation.get("error", ""))
            else:
                st.success(f"Generated with {recommendation.get('provider')} ({recommendation.get('model')}).")
            st.markdown(recommendation["recommendations"])
            st.download_button(
                "Download recommendations",
                data=recommendation["recommendations"].encode("utf-8"),
                file_name="ecotwin_recommendations.md",
                mime="text/markdown",
            )

        st.caption("Review all recommendations with the responsible operator or engineer. Never treat generated advice as automatic control or verified savings.")

# =========================== Device Controls ================================
with devices_tab:
    st.subheader("Facility control room")
    st.warning(
        "These controls only change the in-memory simulation. They do not connect "
        "to real devices or send commands to a facility. Vision can suggest a review; "
        "it never switches physical equipment."
    )

    controller = st.session_state.controller
    all_devices = controller.available_devices
    lighting_devices = [device for device in all_devices if device["kind"] == "lighting"]
    hvac_devices = [device for device in all_devices if device["kind"] == "hvac"]
    lighting_sim_on = any(controller.devices_on.get(device["name"], True) for device in lighting_devices)
    hvac_sim_on = any(controller.devices_on.get(device["name"], True) for device in hvac_devices)
    camera_lights_on = st.session_state.get("cv_lights_on", lighting_sim_on)
    camera_hvac_on = st.session_state.get("cv_hvac_on", hvac_sim_on)
    vision_source = st.session_state.cv_source
    vision_updated_at = st.session_state.cv_updated_at
    vision_is_fresh = (
        vision_source == "Live camera"
        and vision_updated_at is not None
        and (datetime.now() - vision_updated_at).total_seconds() < 10
    )
    vision_has_reading = vision_source == "Demo input" or vision_is_fresh
    vision_count = int(st.session_state.cv_count)
    vision_capacity = int(st.session_state.get("cv_room_capacity", 30))

    st.markdown("#### Latest room signals")
    signal_cols = st.columns(4)
    signal_cols[0].metric(
        "Vision feed",
        "Live" if vision_is_fresh else "Demo" if vision_source == "Demo input" else "Last seen" if vision_source == "Live camera" else "Not started",
    )
    signal_cols[1].metric(
        "People / capacity",
        f"{vision_count} / {vision_capacity}" if vision_has_reading else "No reading",
        f"{min(100, vision_count / vision_capacity * 100):.0f}% occupancy proxy" if vision_has_reading else None,
    )
    signal_cols[2].metric(
        "Entry / exit",
        f"{st.session_state.cv_entries} / {st.session_state.cv_exits}" if vision_is_fresh else "Live only",
    )
    signal_cols[3].metric(
        "Equipment load (simulation)",
        f"{controller.current_power_watts():,.0f} W",
        f"{sum(controller.devices_on.values())} devices on",
    )

    sync_col, state_col = st.columns([1, 2])
    with sync_col:
        if vision_has_reading and controller.occupancy != vision_count:
            if st.button("Copy vision count to simulator", key="sync_cv_to_device_controls"):
                controller.set_occupancy(vision_count)
                st.success("Vision count copied to the Digital Twin simulation.")
                st.rerun()
    with state_col:
        if vision_has_reading:
            room_state = "people detected" if vision_count else "no people detected"
            st.caption(f"Latest source: {vision_source}. The visual occupancy estimate shows {room_state}; verify the room before acting.")
        else:
            st.caption("Start the camera in Live Computer Vision or set a demo count to bring occupancy context into this panel.")

    recommendations = []
    if vision_has_reading and vision_count == 0 and (camera_lights_on or camera_hvac_on):
        recommendations.append("Possible empty-room waste: verify the space, then review lights and HVAC schedules.")
    elif not vision_has_reading and controller.occupancy == 0 and (lighting_sim_on or hvac_sim_on):
        recommendations.append("The Digital Twin simulation is vacant while equipment is on; review the simulated load.")
    if vision_is_fresh and st.session_state.cv_brightness >= 155 and camera_lights_on:
        recommendations.append("Bright-scene cue: check a calibrated light sensor before dimming artificial lighting.")
    if st.session_state.get("cv_opening_status", "Unknown") == "Open" and st.session_state.get("cv_opening_hvac", False):
        recommendations.append("An operator reported an open window/door while HVAC is running; verify and review the operating plan.")
    equipment_findings = st.session_state.get("cv_equipment_findings", [])
    solar_findings = st.session_state.get("cv_solar_findings", [])
    if equipment_findings or (solar_findings and "No visible concern" not in solar_findings):
        recommendations.append("Operator-confirmed inspection findings are waiting for facilities maintenance review.")

    st.markdown("#### Vision review queue")
    if recommendations:
        for recommendation in recommendations:
            st.info(recommendation)
    else:
        st.success("No current vision-informed action is waiting for review.")
    if vision_has_reading and vision_count == 0 and lighting_sim_on:
        confirm_vacancy = st.checkbox(
            "I have confirmed the room is empty; apply lighting change to the simulation only",
            key="confirm_simulated_vacancy",
        )
        if st.button(
            "Switch off non-critical lights in simulation",
            key="apply_vacancy_lighting_simulation",
            disabled=not confirm_vacancy,
        ):
            for device in lighting_devices:
                if not device["critical"]:
                    controller.set_device(device["name"], False)
                    st.session_state[f"device_{environment}_{device['name']}"] = False
            st.success("Non-critical lights switched off in the simulation. No physical device was changed.")
            st.rerun()

    st.markdown("#### Equipment controls")
    category_filter = st.radio(
        "Show equipment",
        ["All", "Lighting", "HVAC", "Other", "Critical"],
        horizontal=True,
        key="device_category_filter",
    )
    if category_filter == "Lighting":
        visible_devices = [device for device in all_devices if device["kind"] == "lighting"]
    elif category_filter == "HVAC":
        visible_devices = [device for device in all_devices if device["kind"] == "hvac"]
    elif category_filter == "Other":
        visible_devices = [device for device in all_devices if device["kind"] not in {"lighting", "hvac"}]
    elif category_filter == "Critical":
        visible_devices = [device for device in all_devices if device["critical"]]
    else:
        visible_devices = all_devices

    for device in visible_devices:
        name = device["name"]
        is_on = controller.devices_on.get(name, True)
        col1, col2, col3 = st.columns([2, 1, 2])
        col1.write(f"**{name}**")
        kind_label = {"lighting": "Lighting", "hvac": "HVAC"}.get(device["kind"], device["kind"].title())
        col1.caption(f"{kind_label} · {'Critical' if device['critical'] else 'Non-critical'}")
        col2.write(f"{device['watts']} W")
        if device["critical"]:
            col3.caption("Confirm safety requirements before changing")
        desired = col3.toggle(
            "On",
            value=is_on,
            key=f"device_{environment}_{name}",
        )
        if desired != is_on:
            controller.set_device(name, desired)

    load_col, energy_col, critical_col = st.columns(3)
    load_col.metric("Simulated power draw", f"{controller.current_power_watts():,.0f} W")
    energy_col.metric("Accumulated simulated energy", f"{controller.energy_used_kwh:.5f} kWh")
    critical_on = sum(
        1 for device in all_devices
        if device["critical"] and controller.devices_on.get(device["name"], True)
    )
    critical_col.metric("Critical devices operating", f"{critical_on} / {sum(device['critical'] for device in all_devices)}")
    st.caption("The displayed power uses illustrative rated watts. Use smart-meter or device telemetry for measured consumption and verified savings.")

# --------------------------- Footer -----------------------------------------
st.divider()
st.caption(
    f"EcoTwin AI prototype · Updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
    "Planning estimates only; not a certified energy audit."
)
