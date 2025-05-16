# app.py
# ---------------------------------------------------------------------
# Streamlit front‑end for the Paddy Disease / Variety / Age predictor
# compatible with the unified inference bundle (model_bundle.pkl)
# ---------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import time, base64, os
from datetime import datetime
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Direct import from inference.py (no need for utils package)
from utils.inference import predict, _labels, _load_bundle

# --------------------------------------------------------------------
# Page config & CSS
# --------------------------------------------------------------------
st.set_page_config(
    page_title="Paddy Predictor – COSC2753 A2",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-header {font-size:2.5rem;color:#2e7d32;margin-bottom:0;}
        .sub-header {font-size:1.1rem;color:#5c5c5c;margin-top:0;margin-bottom:2rem;}
        .stButton>button {background:#2e7d32;color:white;border-radius:4px;padding:0.5rem 1rem;font-weight:bold;}
        .stButton>button:hover {background:#1b5e20;border-color:#1b5e20;}
        .stat-card {background:#f1f8e9;border-radius:10px;padding:1.5rem;text-align:center;
                    box-shadow:0 4px 6px rgba(0,0,0,0.1);}
        .stat-value {font-size:2rem;font-weight:bold;color:#2e7d32;}
        .stat-label {font-size:1rem;color:#5c5c5c;}
        .model-selector {margin:1.5rem 0;padding:1rem;background:#f9f9f9;
                         border-radius:8px;border-left:4px solid #2e7d32;}
        .footer {text-align:center;margin-top:3rem;padding:1rem;background:#274e13;border-radius:8px;color:white;}
        .prediction-card {background:white;border-radius:8px;padding:1rem;margin-bottom:1rem;
                          box-shadow:0 2px 4px rgba(0,0,0,0.1);}
        .confidence-high {color:#007500;}
        .confidence-medium {color:#FFA500;}
        .confidence-low {color:#FF0000;}
        .metric-container {display:flex;flex-direction:column;background:white;border-radius:8px;
                           padding:1rem;box-shadow:0 2px 4px rgba(0,0,0,0.1);height:100%;}
        .metric-title {font-size:1rem;color:#5c5c5c;margin-bottom:0.5rem;}
        .metric-value {font-size:1.8rem;font-weight:bold;color:#2e7d32;}
        .metric-delta {font-size:0.9rem;margin-top:0.3rem;}
        .model-info {background:#f1f8e9;border-radius:8px;padding:12px;margin-bottom:10px;}
        .error-message {color:#d32f2f;background:#ffebee;padding:10px;border-radius:4px;margin:10px 0;}
        .compatibility-message {background:#fff3e0;border-radius:4px;padding:10px;margin:10px 0;border-left:4px solid #ff9800;}
        .disabled-option {color:#9e9e9e;cursor:not-allowed;}
        .tooltip {position:relative;display:inline-block;}
        .tooltip .tooltiptext {visibility:hidden;width:200px;background-color:#555;color:#fff;text-align:center;
                               border-radius:6px;padding:5px;position:absolute;z-index:1;bottom:125%;left:50%;
                               margin-left:-100px;opacity:0;transition:opacity 0.3s;}
        .tooltip:hover .tooltiptext {visibility:visible;opacity:1;}
        .help-icon {color:#9e9e9e;font-size:14px;cursor:help;}
        .algorithm-compatibility {font-size:0.85rem;color:#5c5c5c;margin-top:5px;padding:5px;
                                 background:#f5f5f5;border-radius:4px;}
        .badge {display:inline-block;padding:0.25em 0.4em;font-size:75%;font-weight:700;line-height:1;
               text-align:center;white-space:nowrap;vertical-align:baseline;border-radius:0.25rem;}
        .badge-success {background-color:#c8e6c9;color:#2e7d32;}
        .badge-warning {background-color:#ffecb3;color:#ff8f00;}
        .badge-danger {background-color:#ffcdd2;color:#c62828;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------- #
# Check if model file exists
# -------------------------------------------------- #
def check_model_availability():
    """Check if the model bundle file exists at the expected location"""
    model_path = "models/full_model_bundle.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}!")
        st.info(
            """
            Please ensure:
            1. You have the model bundle file 'full_model_bundle.pkl'
            2. It's placed in a directory called 'models' in the same folder as this app
            3. The file name matches exactly 'full_model_bundle.pkl'
            """
        )
        return False
    return True


# -------------------------------------------------- #
# Helper utilities
# -------------------------------------------------- #
def create_confidence_badge(confidence: float) -> str:
    """Create an HTML badge for confidence display"""
    if confidence >= 0.8:
        return f'<span class="confidence-high">{confidence:.1%}</span>'
    if confidence >= 0.5:
        return f'<span class="confidence-medium">{confidence:.1%}</span>'
    return f'<span class="confidence-low">{confidence:.1%}</span>'


def image_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string for HTML display"""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def create_custom_metric(label, value, delta=None, delta_color="normal") -> str:
    """Create an HTML metric component"""
    html = (
        f'<div class="metric-container"><div class="metric-title">{label}</div>'
        f'<div class="metric-value">{value}</div>'
    )
    if delta:
        colour = "green" if delta_color == "normal" else "red"
        html += f'<div class="metric-delta" style="color:{colour}">{delta}</div>'
    return html + "</div>"


# Simple function to load an image
def load_image(file):
    """Load and validate an image file"""
    try:
        img = Image.open(file)
        img = img.convert("RGB")  # Ensure RGB format
        return img, np.array(img)
    except Exception as e:
        st.error(f"Error loading image {file.name}: {str(e)}")
        return None, None


# -------------------------------------------------- #
# Get available model algorithms
# -------------------------------------------------- #
@st.cache_resource
def get_available_algos():
    """Retrieve available model algorithms from the bundle"""
    try:
        bundle = _load_bundle()
        # Get keys from disease classifier as reference
        algos = list(bundle["models"]["disease_classifier"].keys())

        # Map internal keys to display names
        algo_map = {
            "lgb": "LightGBM",
            "lightgbm": "LightGBM",
            "rf": "Random Forest",
            "random_forest": "Random Forest",
            "knn": "k-NN"
        }

        # Create a display map and reverse map
        display_algos = [algo_map.get(a, a.title()) for a in algos]
        reverse_map = {algo_map.get(a, a.title()): a for a in algos}

        return display_algos, reverse_map
    except Exception as e:
        st.error(f"Failed to load model algorithms: {str(e)}")
        return ["LightGBM", "Random Forest", "k-NN"], {
            "LightGBM": "lgb",
            "Random Forest": "rf",
            "k-NN": "knn"
        }


# -------------------------------------------------- #
# Algorithm-Task compatibility mapping
# -------------------------------------------------- #
def get_algorithm_task_compatibility():
    """Returns a dictionary mapping algorithms to compatible tasks"""
    return {
        "LightGBM": ["disease", "variety", "age"],
        "Random Forest": ["disease", "variety"],
        "k-NN": ["disease", "variety", "age"]
    }


# -------------------------------------------------- #
# Header
# -------------------------------------------------- #
cols = st.columns([6, 1])
with cols[0]:
    st.markdown('<h1 class="main-header">🌾 Paddy Predictor - COSC 2752 | COSC 2812 - S2_G4 Group</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Group Assignment: AI-based disease, variety & age analysis for rice leaves</p>',
        unsafe_allow_html=True,
    )

# Check if model is available
model_available = check_model_availability()

# Get available algorithms
if model_available:
    available_algos, algo_map = get_available_algos()
else:
    available_algos = ["LightGBM", "Random Forest", "k-NN"]
    algo_map = {"LightGBM": "lgb", "Random Forest": "rf", "k-NN": "knn"}

# Algorithm compatibility map
algo_compatibility = get_algorithm_task_compatibility()

# -------------------------------------------------- #
# Sidebar – configuration
# -------------------------------------------------- #
with st.sidebar:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRjzzvkiuQj7icQk1wm2y92cs0VgWVsrfAIhA&s",
        width=300,
    )
    st.header("Analysis Configuration")

    # Initialize session state variables for tracking selections
    if "current_task" not in st.session_state:
        st.session_state.current_task = "All tasks"

    if "current_algorithm" not in st.session_state:
        st.session_state.current_algorithm = "LightGBM"

    if "compatibility_warning" not in st.session_state:
        st.session_state.compatibility_warning = False

    # Task selection first - affects algorithm availability
    task_choice = st.radio(
        "Select task(s) to run:",
        ("All tasks", "Disease only", "Variety only", "Age only"),
        key="task_selector",
    )

    # Update task selection in session state
    st.session_state.current_task = task_choice

    # Define tasks mapping
    TASKS_MAP = {
        "All tasks": {"disease", "variety", "age"},
        "Disease only": {"disease"},
        "Variety only": {"variety"},
        "Age only": {"age"},
    }
    tasks_to_run = TASKS_MAP[task_choice]

    # Display algorithm compatibility information based on task
    st.markdown("### Algorithm Selection")

    # Filter available algorithms based on task compatibility
    compatible_algos = []
    for algo in available_algos:
        is_compatible = all(task in algo_compatibility.get(algo, []) for task in tasks_to_run)
        if is_compatible:
            compatible_algos.append(algo)

    # Show compatibility badges for each algorithm
    for algo in available_algos:
        is_compatible = all(task in algo_compatibility.get(algo, []) for task in tasks_to_run)
        badge_class = "success" if is_compatible else "danger"
        badge_text = "Compatible" if is_compatible else "Incompatible"
        st.markdown(
            f"<div><b>{algo}</b>: <span class='badge badge-{badge_class}'>{badge_text}</span></div>",
            unsafe_allow_html=True
        )

    # Algorithm family selection → must match keys in model_bundle.pkl
    model_choice = st.radio(
        "Select algorithm family:",
        options=available_algos,
        index=0 if st.session_state.current_algorithm in available_algos else available_algos.index("LightGBM")
    )

    # Check if the selected model is compatible with the task
    is_compatible = all(task in algo_compatibility.get(model_choice, []) for task in tasks_to_run)

    # Show warning if incompatible selection
    if not is_compatible:
        st.markdown(
            f"""
            <div style="color:#FFFFFF;">
                ⚠️ <b>{model_choice}</b> is not compatible with <b>{task_choice}</b>.<br>
                {model_choice} supports: {', '.join(algo_compatibility.get(model_choice, []))}
            </div>
            """,
            unsafe_allow_html=True
        )
        st.session_state.compatibility_warning = True
    else:
        st.session_state.compatibility_warning = False
        # Only update algorithm if compatible
        st.session_state.current_algorithm = model_choice
        selected_algo = algo_map[model_choice]

    # Batch settings
    st.subheader("Batch Settings")
    batch_size = st.slider("Max images to process", 1, 50, value=20)

    # Advanced options
    with st.expander("Advanced Options"):
        threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
        show_thumbnails = st.toggle("Show image thumbnails", value=True)
        enable_charts = st.toggle("Show prediction charts", value=True)
        show_raw_features = st.toggle("Show raw features", value=False)

    # Model information
    if model_available:
        with st.expander("Model Information"):
            try:
                bundle = _load_bundle()

                # Get available labels for disease and variety
                disease_classes = ", ".join(_labels("disease", bundle))
                variety_classes = ", ".join(_labels("variety", bundle))

                st.markdown(f"""
                <div class="model-info">
                <strong>Model Bundle:</strong> full_model_bundle.pkl<br>
                <strong>Disease Classes:</strong> {disease_classes}<br>
                <strong>Variety Classes:</strong> {variety_classes}<br>
                <strong>Available Algorithms:</strong> {", ".join(available_algos)}<br>
                <strong>Algorithm Capabilities:</strong><br>
                - LightGBM: All tasks (disease, variety, age)<br>
                - RandomForest: Disease, variety only<br>
                - k-NN: All tasks
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading model info: {str(e)}")

    # Help
    with st.expander("Need Help?"):
        st.markdown(
            """
            **Usage steps**
            1. Upload paddy leaf images (JPG/PNG).  
            2. Select task(s) to analyze.  
            3. Select a compatible algorithm. Note that:
               - LightGBM supports all tasks
               - Random Forest supports disease and variety only
               - k-NN supports all tasks
            4. Click **Run Analysis**.  
            5. Inspect & download results.

            Contact: support@paddypredictor.com
            """
        )

# -------------------------------------------------- #
# File uploader
# -------------------------------------------------- #
st.subheader("1. Upload Images")
uploader_cols = st.columns([3, 1])
with uploader_cols[0]:
    uploaded_files = st.file_uploader(
        f"Upload up to {batch_size} images (≤ 10 MB each)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
with uploader_cols[1]:
    if uploaded_files:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-value">{len(uploaded_files)}</div>
                <div class="stat-label">File(s) ready</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------------------------------------- #
# Configuration summary and Run button
# -------------------------------------------------- #
st.subheader("2. Run Analysis")

# Create a clear configuration summary
# First display the configuration summary in full width
st.markdown("#### Current Configuration")

# Determine if current configuration is valid
current_task = st.session_state.current_task
current_algorithm = st.session_state.current_algorithm

tasks_list = list(TASKS_MAP[current_task])
task_display = ", ".join([t.capitalize() for t in tasks_list])

# Check compatibility
is_valid_config = all(task in algo_compatibility.get(current_algorithm, []) for task in TASKS_MAP[current_task])

# Display configuration summary with status
st.markdown(
    f"""
    <div style="padding:15px;background:{'#73946B' if is_valid_config else '#ffebee'};border-radius:8px;margin-bottom:15px;">
        <div><b>Task(s):</b> {current_task}</div>
        <div><b>Algorithm:</b> {current_algorithm}</div>
        <div><b>Status:</b> {'✅ Ready to run' if is_valid_config else '❌ Incompatible selection'}</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Run analysis button - placed below the configuration summary
run_analysis = st.button(
    "🚀 Run Analysis",
    use_container_width=True,
    disabled=not uploaded_files or not model_available or not is_valid_config or st.session_state.compatibility_warning
)

if st.session_state.compatibility_warning:
    st.markdown(
        """
        <div style="color:#d32f2f;font-size:0.9rem;margin-top:5px;">
        Please select a compatible algorithm for your task(s).
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------- #
# Session state prep
# -------------------------------------------------- #
if "results" not in st.session_state:
    st.session_state.results = None
if "analysis_timestamp" not in st.session_state:
    st.session_state.analysis_timestamp = None
if "extracted_features" not in st.session_state:
    st.session_state.extracted_features = None

# -------------------------------------------------- #
# Processing loop
# -------------------------------------------------- #
if uploaded_files and run_analysis and model_available and not st.session_state.compatibility_warning:
    # Use the selected algorithm from session state to ensure compatibility
    selected_algo = algo_map[st.session_state.current_algorithm]
    tasks_to_run = TASKS_MAP[st.session_state.current_task]

    with st.spinner("Processing images…"):
        prog = st.progress(0)
        status = st.empty()
        results = []
        extracted_features = []

        try:
            for i, file in enumerate(uploaded_files[: batch_size], 1):
                status.text(f"Analyzing {file.name} ({i}/{len(uploaded_files[:batch_size])})")
                pil_img, _ = load_image(file)

                if pil_img is None:
                    continue

                # Call predict with the appropriate algorithm
                preds = predict(pil_img, tasks_to_run, algo=selected_algo)

                # Optionally extract features for display
                if show_raw_features:
                    from utils.inference import _extract_features

                    feats = _extract_features(pil_img)
                    extracted_features.append({
                        "filename": file.name,
                        "features": feats
                    })

                img_str = image_to_base64(pil_img)
                row = {"filename": file.name, "image_data": img_str}

                if "disease" in preds:
                    row.update(
                        {
                            "disease": preds["disease"][0],
                            "disease_conf": preds["disease"][1],
                        }
                    )
                if "variety" in preds:
                    row.update(
                        {
                            "variety": preds["variety"][0],
                            "variety_conf": preds["variety"][1],
                        }
                    )
                if "age" in preds:
                    row["age_days"] = preds["age"]

                results.append(row)
                prog.progress(i / len(uploaded_files[:batch_size]))
                time.sleep(0.1)

            st.session_state.results = results
            if extracted_features:
                st.session_state.extracted_features = extracted_features
            st.session_state.analysis_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status.empty()
            time.sleep(0.3)
            prog.empty()
            st.success(f"Finished! Processed {len(results)} image(s).")
            st.rerun()
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback

            error_details_raw = traceback.format_exc()
            # Perform the replacement outside the f-string
            error_details_html = error_details_raw.replace('\n', '<br>')

            st.markdown(f"""
                <div class="error-message">
                <strong>Error details:</strong><br>
                {error_details_html}
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------- #
# Results display
# -------------------------------------------------- #
if st.session_state.results:
    results = st.session_state.results
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "image_data"} for r in results])

    st.header("3. Analysis Results")
    st.subheader(f"Completed at {st.session_state.analysis_timestamp}")

    # Quick metrics
    if len(df) > 0:
        metric_cols = st.columns(3)

        if "disease" in df.columns:
            with metric_cols[0]:
                counts = df["disease"].value_counts()
                if not counts.empty:
                    top = counts.index[0]
                    st.markdown(
                        create_custom_metric("Most common disease", top, f"{counts[0] / len(df):.1%}"),
                        unsafe_allow_html=True,
                    )

        if "variety" in df.columns:
            with metric_cols[1]:
                counts = df["variety"].value_counts()
                if not counts.empty:
                    top = counts.index[0]
                    st.markdown(
                        create_custom_metric("Most common variety", top, f"{counts[0] / len(df):.1%}"),
                        unsafe_allow_html=True,
                    )

        if "age_days" in df.columns:
            with metric_cols[2]:
                avg_age = df["age_days"].mean()
                st.markdown(
                    create_custom_metric(
                        "Average age", f"{avg_age:.1f} days",
                        f"Range: {df['age_days'].min():.1f}–{df['age_days'].max():.1f}",
                    ),
                    unsafe_allow_html=True,
                )

    # Tabs for different result views
    tab_list = ["Data Table", "Visualisations", "Image Gallery"]
    if st.session_state.extracted_features:
        tab_list.append("Raw Features")

    tabs = st.tabs(tab_list)

    # --- tab1: table + downloads
    with tabs[0]:
        view_df = df.copy()
        if "disease_conf" in view_df.columns:
            view_df["disease_confidence"] = view_df["disease_conf"].apply(lambda x: f"{x:.1%}")
            view_df = view_df.drop("disease_conf", axis=1)
        if "variety_conf" in view_df.columns:
            view_df["variety_confidence"] = view_df["variety_conf"].apply(lambda x: f"{x:.1%}")
            view_df = view_df.drop("variety_conf", axis=1)

        st.dataframe(view_df, use_container_width=True)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False).encode(),
                f"paddy_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv",
            )
        with d2:
            xls_buf = BytesIO()
            df.to_excel(xls_buf, index=False)
            st.download_button(
                "📊 Download Excel",
                xls_buf.getvalue(),
                f"paddy_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                "application/vnd.ms-excel",
            )

    # --- tab2: charts
    with tabs[1]:
        if enable_charts:
            chart_cols = st.columns(2)
            if "disease" in df.columns:
                with chart_cols[0]:
                    disease_counts = df["disease"].value_counts().reset_index()
                    disease_counts.columns = ["Disease", "Count"]

                    fig = px.bar(
                        disease_counts,
                        x="Disease",
                        y="Count",
                        title="Disease distribution",
                        color="Count",
                        color_continuous_scale="greens",
                    )
                    fig.update_layout(height=400, xaxis_title="Disease", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
            if "variety" in df.columns:
                with chart_cols[1]:
                    variety_counts = df["variety"].value_counts().reset_index()
                    variety_counts.columns = ["Variety", "Count"]

                    fig = px.pie(
                        variety_counts,
                        names="Variety",
                        values="Count",
                        title="Variety distribution",
                        color_discrete_sequence=px.colors.sequential.Greens,
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            if "age_days" in df.columns:
                fig = px.histogram(
                    df, x="age_days", nbins=10, title="Age distribution", color_discrete_sequence=["#2e7d32"]
                )
                fig.update_layout(height=400, xaxis_title="Age (days)")
                st.plotly_chart(fig, use_container_width=True)
            if "disease_conf" in df.columns or "variety_conf" in df.columns:
                st.subheader("Prediction confidence")
                gauge_cols = st.columns(2)
                if "disease_conf" in df.columns:
                    with gauge_cols[0]:
                        avg_conf = df["disease_conf"].mean()
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=avg_conf,
                                title={"text": "Avg disease confidence"},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": "#2e7d32"},
                                    "steps": [
                                        {"range": [0, 0.5], "color": "#ffcdd2"},
                                        {"range": [0.5, 0.8], "color": "#ffe0b2"},
                                        {"range": [0.8, 1], "color": "#c8e6c9"},
                                    ],
                                },
                            )
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                if "variety_conf" in df.columns:
                    with gauge_cols[1]:
                        avg_conf = df["variety_conf"].mean()
                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=avg_conf,
                                title={"text": "Avg variety confidence"},
                                gauge={
                                    "axis": {"range": [0, 1]},
                                    "bar": {"color": "#2e7d32"},
                                    "steps": [
                                        {"range": [0, 0.5], "color": "#ffcdd2"},
                                        {"range": [0.5, 0.8], "color": "#ffe0b2"},
                                        {"range": [0.8, 1], "color": "#c8e6c9"},
                                    ],
                                },
                            )
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
    # --- tab3: gallery
    with tabs[2]:
        if show_thumbnails:
            st.subheader("Image gallery")
            gallery_cols = st.columns(4)


            # ── helper: coloured confidence badge ───────────────────────────────
            def badge(conf: float) -> str:
                pct = conf * 100
                colour = "#198754" if pct >= 90 else "#ffc107" if pct >= 75 else "#dc3545"
                return (f'<span style="background:{colour};color:#fff;'
                        f'padding:6px 6px 6px 6px;border-radius:4px;font-weight:600;'
                        f'font-size:0.75rem;margin-left:4px;">{pct:.1f}%</span>')


            # -------------------------------------------------------------------

            for i, r in enumerate(results):
                with gallery_cols[i % 4]:
                    html = f'''
                    <div style="background:#fff;border-radius:10px;padding:12px 12px 12px 12px;
                                box-shadow:0 2px 8px rgba(0,0,0,0.15); margin-bottom:20px;">
                      <img src="data:image/jpeg;base64,{r["image_data"]}"
                           style="width:100%;border-radius:8px;margin-bottom:20px;">
                      <div style="font-weight:700;font-size:0.95rem;color:#000;
                                  overflow:hidden;text-overflow:ellipsis;
                                  white-space:nowrap;margin-bottom:6px;">
                          {r["filename"]}
                      </div>
                    '''

                    if "disease" in r:
                        html += (f'<div style="font-size:0.85rem;color:#1a1a1a;'
                                 f'margin-bottom:4px;"><b>Disease:</b> {r["disease"]} '
                                 f'{badge(r["disease_conf"])}</div>')
                    if "variety" in r:
                        html += (f'<div style="font-size:0.85rem;color:#1a1a1a;'
                                 f'margin-bottom:4px;"><b>Variety:</b> {r["variety"]} '
                                 f'{badge(r["variety_conf"])}</div>')
                    if "age_days" in r:
                        html += (f'<div style="font-size:0.85rem;color:#1a1a1a;">'
                                 f'<b>Age:</b> {r["age_days"]:.1f} days</div>')

                    st.markdown(html + "</div>", unsafe_allow_html=True)

    # --- tab4: raw features (optional)
    if len(tab_list) > 3 and st.session_state.extracted_features:
        with tabs[3]:
            st.subheader("Extracted Image Features")
            st.write("These are the raw features extracted from each image and used for prediction:")

            # Select an image to view features
            feature_files = [item["filename"] for item in st.session_state.extracted_features]
            selected_file = st.selectbox("Select image to view features", feature_files)

            # Show features for selected image
            for item in st.session_state.extracted_features:
                if item["filename"] == selected_file:
                    # Display feature dataframe
                    st.dataframe(item["features"], use_container_width=True)

                    # Visualize important features
                    st.subheader("Feature visualization")
                    feat_viz_cols = st.columns(2)

                    with feat_viz_cols[0]:
                        # Color channel statistics
                        color_stats = item["features"][["mean_r", "mean_g", "mean_b",
                                                        "std_r", "std_g", "std_b"]].melt()
                        color_stats.columns = ["Feature", "Value"]

                        fig = px.bar(color_stats, x="Feature", y="Value",
                                     color="Feature",
                                     title="RGB Channel Statistics",
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                        st.plotly_chart(fig, use_container_width=True)

                    with feat_viz_cols[1]:
                        # Create histogram visualization of R, G, B histograms from features
                        r_hist = [item["features"][f"r_hist_{i}"].values[0] for i in range(8)]
                        g_hist = [item["features"][f"g_hist_{i}"].values[0] for i in range(8)]
                        b_hist = [item["features"][f"b_hist_{i}"].values[0] for i in range(8)]

                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=list(range(8)), y=r_hist, name='Red', marker_color='red', opacity=0.7))
                        fig.add_trace(
                            go.Bar(x=list(range(8)), y=g_hist, name='Green', marker_color='green', opacity=0.7))
                        fig.add_trace(go.Bar(x=list(range(8)), y=b_hist, name='Blue', marker_color='blue', opacity=0.7))

                        fig.update_layout(title='RGB Histograms',
                                          xaxis_title='Bin',
                                          yaxis_title='Normalized Frequency',
                                          barmode='group')
                        st.plotly_chart(fig, use_container_width=True)

                    # Download features
                    st.download_button(
                        "📥 Download Features CSV",
                        item["features"].to_csv(index=False).encode(),
                        f"{selected_file.split('.')[0]}_features.csv",
                        "text/csv"
                    )
                    break

# -------------------------------------------------- #
# Model and project info
# -------------------------------------------------- #
st.markdown("---")
info_cols = st.columns(2)

with info_cols[0]:
    with st.expander("Model information"):
        st.markdown(
            """
            ### Algorithm architectures in this bundle

            **Feature-based models:**
            - **LightGBM** – gradient‑boosted decision trees; strong baseline across all tasks.  
            - **Random Forest** – ensemble of trees; good interpretability.  
            - **k‑NN** – tuned for age estimation; offers non‑parametric regression.

            ### Validation snapshot  

            | Task    | Metric | LightGBM | Random Forest | k‑NN |
            |---------|--------|----------|---------------|------|
            | Disease | F1     | 0.89     | 0.91          | —    |
            | Variety | Acc    | 0.87     | 0.85          | —    |
            | Age     | RMSE(d)| 3.2      | 3.8           | 2.9  |
            """
        )

with info_cols[1]:
    with st.expander("About this project", expanded=True):
        st.markdown(
            """
            # Paddy Predictor

            Advanced ML system for paddy leaf analysis, built for **COSC2753 Assignment 2**.

            ### Team
            **Emma** – Main Model-researcher, Report
            
            **Mẫn** – Main Model-researcher, Report
            
            **Hà** – Main Model-researcher, Report
            
            **Hào** – Sub Model-researcher, Report, Data Pre-processing
            
            **Quân** – Sub Model-researcher, Report, EDA, Website

            © 2025 RMIT University
            """,
            unsafe_allow_html=True
        )

# Footer
st.markdown(
    """
    <div class="footer">
        🌾 Paddy Predictor v1.0 | Made with ❤️ from [RMIT 2024 S3] COSC 2752 | COSC 2812 - S2_G4 Group Members| 
        <a href="s3927181@rmit.edu.vn" style="color:#fff;">Contact Support</a>
    </div>
    """,
    unsafe_allow_html=True,
)