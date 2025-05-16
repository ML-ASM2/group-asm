"""
inference.py – single-file solution
-----------------------------------
Drop this into your Streamlit / FastAPI backend.
It loads *one* pickle ("full_model_bundle.pkl") that you exported from the
notebook and provides:

    predict(pil_image,
            tasks={"disease", "variety", "age"},
            algo="lgb")  # or "rf", "knn", ...

Returns
-------
dict  – {
           "disease": (label, confidence),   # if requested
           "variety": (label, confidence),   # if requested
           "age":     age_in_days            # if requested
        }
"""

# ═════════ STANDARD & THIRD-PARTY LIBS ═════════
import os
from typing import Dict, Iterable, Set, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import iqr, skew, kurtosis
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy

# Optional – will silently do nothing outside Streamlit
try:
    import streamlit as st
    _cache = st.cache_resource
    _warn  = st.warning
    _err   = st.error
except ModuleNotFoundError:  # plain Python script
    def _cache(**_):         # dummy decorators
        def deco(f): return f
        return deco
    _warn = _err = lambda *_: None

# ===================== Custom ensemble (needed for un-pickling) =====================

class LGB_KNN_Ensemble:
    """
    Minimal re-implementation so that joblib can un-pickle the object
    stored as 'task3_ensemble' in your bundle.

    The original train notebook used:
        return 0.5 * lgb.predict(X) + 0.5 * knn.predict(X)

    We reproduce that behaviour here.  If your training code
    changed the weights, adjust them below.
    """
    def __init__(self, lgb_model, knn_model, w_lgb: float = 0.5, w_knn: float = 0.5):
        self.lgb_model = lgb_model
        self.knn_model = knn_model
        self.w_lgb = w_lgb
        self.w_knn = w_knn

    # -------- duck-typing; only predict() is required during inference ----------
    def predict(self, X):
        return (
            self.w_lgb * self.lgb_model.predict(X) +
            self.w_knn * self.knn_model.predict(X)
        )

# ================================================================================
# --- 1. Declare the class (same as you already added) ----------------
class LGB_KNN_Ensemble:
    def __init__(self, lgb_model, knn_model, w_lgb=0.5, w_knn=0.5):
        self.lgb_model, self.knn_model = lgb_model, knn_model
        self.w_lgb, self.w_knn = w_lgb, w_knn
    def predict(self, X):
        return self.w_lgb * self.lgb_model.predict(X) + self.w_knn * self.knn_model.predict(X)

# --- 2. **Register** the symbol under the name the pickle expects ----
import sys
sys.modules["__main__"].LGB_KNN_Ensemble = LGB_KNN_Ensemble
# --------------------------------------------------------------------

# ═════════════════════ CONFIG ═══════════════════
MODEL_BUNDLE_PATH = "models/full_model_bundle.pkl"  # update if moved
HIST_BINS = 8
LBP_P, LBP_R = 8, 1.0
# ════════════════════════════════════════════════

# Public-task → internal-dict key
TASK_KEY = {
    "disease": "disease_classifier",
    "variety": "variety_classifier",
    "age":     "age_regressor",
}

DEFAULT_DISEASE_NAMES = [
    "bacterial_leaf_blight", "bacterial_leaf_streak", "bacterial_panicle_blight",
    "blast", "brown_spot", "dead_heart", "downy_mildew",
    "hispa", "normal", "tungro",
]
DEFAULT_VARIETY_NAMES = [
    "ADT45", "AndraPonni", "AtchayaPonni", "IR20", "KarnatakaPonni",
    "Onthanel", "Ponni", "RR", "Surya", "Zonal",
]

# ═════════════════════ BUNDLE LOADER ═════════════════════
@_cache(show_spinner="🔄 Loading model bundle…")
def _load_bundle() -> Dict:
    raw = joblib.load(MODEL_BUNDLE_PATH)

    # ---- 1) Model dict (normalize flat → nested) ----
    models = {}
    if "models" in raw:
        models = raw["models"]       # already nested
    else:
        models = {
            "disease_classifier": {},
            "variety_classifier": {},
            "age_regressor":      {},
        }
        for k, v in raw.items():
            if   k.startswith("task1_"): models["disease_classifier"][k[6:]] = v
            elif k.startswith("task2_"): models["variety_classifier"][k[6:]] = v
            elif k.startswith("task3_"): models["age_regressor"][k[6:]]      = v

    # ---- 2) Encoders ----
    enc = {}
    if "le_label"   in raw: enc["disease"] = raw["le_label"]
    if "le_variety" in raw: enc["variety"] = raw["le_variety"]

    # ---- 3) Feature lists per task ----
    def _hist_feats(ch): return [f"{ch}_hist_{i}" for i in range(HIST_BINS)]
    full_union = [
        # base stats
        "mean_r","mean_g","mean_b","std_r","std_g","std_b",
        "median_r","median_g","median_b","iqr_r","iqr_g","iqr_b",
        "r_g_ratio","g_b_ratio","r_b_ratio",
        # optional extras
        "entropy","edge_density",
        "skew_r","skew_g","skew_b","kurt_r","kurt_g","kurt_b",
        "lbp_mean","lbp_std",
        # histograms
        *_hist_feats("r"), *_hist_feats("g"), *_hist_feats("b"),
    ]

    # Try to retrieve precise lists from pickle (they were saved as base_feats_*)
    task1_cols = raw.get("base_feats_disease", []) + \
                 raw.get("optional_feats", []) + _hist_feats("r") + \
                 _hist_feats("g") + _hist_feats("b")
    task2_cols = raw.get("base_feats_variety", []) + \
                 raw.get("optional_feats", []) + _hist_feats("r") + \
                 _hist_feats("g") + _hist_feats("b")
    task3_cols = raw.get("base_feats_reg", []) + \
                 raw.get("optional_feats", []) + _hist_feats("r") + \
                 _hist_feats("g") + _hist_feats("b")

    # Fallback to union if any list came out empty
    if not task1_cols: task1_cols = full_union
    if not task2_cols: task2_cols = full_union
    if not task3_cols: task3_cols = full_union

    feature_map = {
        "task1": task1_cols,
        "task2": task2_cols,
        "task3": task3_cols,
    }

    return {"models": models, "encoders": enc, "features": feature_map}

# ═══════════════ FEATURE EXTRACTION (59 cols) ══════════════
def _extract_features(img_pil: Image.Image) -> pd.DataFrame:
    img = np.array(img_pil).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img]*3, -1)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    std_r, std_g, std_b    = r.std(),  g.std(),  b.std()
    median_r, median_g, median_b = np.median(r), np.median(g), np.median(b)
    iqr_r_val, iqr_g_val, iqr_b_val = iqr(r, axis=None), iqr(g, axis=None), iqr(b, axis=None)
    r_g_ratio = mean_r / (mean_g + 1e-5)
    g_b_ratio = mean_g / (mean_b + 1e-5)
    r_b_ratio = mean_r / (mean_b + 1e-5)

    entropy_val = float(shannon_entropy(img))

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    edge_density = float((np.abs(sobel) > 50).mean())

    skew_r_val,  skew_g_val,  skew_b_val  = skew(r.flat), skew(g.flat), skew(b.flat)
    kurt_r_val, kurt_g_val, kurt_b_val = kurtosis(r.flat), kurtosis(g.flat), kurtosis(b.flat)

    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method="uniform")
    lbp_mean, lbp_std = lbp.mean(), lbp.std()

    # Histograms
    hist_feats = {}
    for ch_arr, name in zip([r, g, b], ["r", "g", "b"]):
        hist = np.histogram(ch_arr, bins=HIST_BINS, range=(0, 256), density=True)[0]
        hist_feats.update({f"{name}_hist_{i}": v for i, v in enumerate(hist)})

    feats = {
        "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b,
        "std_r": std_r,   "std_g": std_g,   "std_b": std_b,
        "median_r": median_r, "median_g": median_g, "median_b": median_b,
        "iqr_r": iqr_r_val, "iqr_g": iqr_g_val, "iqr_b": iqr_b_val,
        "r_g_ratio": r_g_ratio, "g_b_ratio": g_b_ratio, "r_b_ratio": r_b_ratio,
        "entropy": entropy_val, "edge_density": edge_density,
        "skew_r": skew_r_val, "skew_g": skew_g_val, "skew_b": skew_b_val,
        "kurt_r": kurt_r_val, "kurt_g": kurt_g_val, "kurt_b": kurt_b_val,
        "lbp_mean": lbp_mean, "lbp_std": lbp_std,
        **hist_feats,
    }

    # Canonical column order
    all_cols = _load_bundle()["features"]["task1"]  # any task gives full list
    return pd.DataFrame([{c: feats.get(c, 0.0) for c in all_cols}])

# ═══════════════ HELPER FUNCTIONS ═══════════════
def _select_model(task: str, algo: str, bundle_models: Dict):
    pool = bundle_models[TASK_KEY[task]]
    if algo in pool:
        return pool[algo]
    fallback, model = next(iter(pool.items()))
    _warn(f"⚠️ '{algo}' not found for {task}. Using '{fallback}'.")
    return model

def _labels(task: str, bundle) -> Iterable[str]:
    enc = bundle["encoders"].get(task)
    if enc is not None and hasattr(enc, "classes_"):
        return enc.classes_
    return DEFAULT_DISEASE_NAMES if task == "disease" else DEFAULT_VARIETY_NAMES

def _classify(model, feats, labels) -> Tuple[str, float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        return str(labels[idx]), float(probs[idx])
    idx = int(model.predict(feats)[0])
    return str(labels[idx]), 1.0

# ═════════════════════ PUBLIC API ════════════════════
def predict(
    pil_img: Image.Image,
    tasks: Set[str],
    algo: str = "lgb",
) -> Dict:
    """
    Parameters
    ----------
    pil_img : PIL.Image.Image
    tasks   : subset of {"disease", "variety", "age"}
    algo    : key of algorithm to use ("lgb", "rf", "knn", ...)

    Returns
    -------
    dict { task : (label, conf) | age_float }
    """
    tasks = {t.lower() for t in tasks}
    # Assuming TASK_KEY is a dictionary with relevant keys
    unknown = tasks - TASK_KEY.keys()
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}")

    bundle = _load_bundle()
    feats_full = _extract_features(pil_img)
    results = {}

    for task in tasks:
        try:
            cols = bundle["features"]["task1" if task == "disease" else
                                     "task2" if task == "variety" else
                                     "task3"]
            feat_sub = feats_full[cols]

            model = _select_model(task, algo, bundle["models"])

            # >>> align columns to the model
            expected_cols = None
            if hasattr(model, "feature_names_in_"):
                expected_cols = list(model.feature_names_in_)
            elif hasattr(model, "feature_names_"):
                expected_cols = list(model.feature_names_)
            if expected_cols is not None:
                feat_sub = feat_sub.reindex(columns=expected_cols, fill_value=0.0)
            # <<<

            if task in {"disease", "variety"}:
                label, conf = _classify(model, feat_sub, _labels(task, bundle))
                results[task] = (label, conf)
            else: # Assuming 'age' is the other task
                results[task] = float(model.predict(feat_sub)[0])

        except Exception as e:
            _err(f"❌ Error predicting {task}: {e}")
            if task in {"disease", "variety"}:
                fallback = "normal" if task == "disease" else DEFAULT_VARIETY_NAMES[0]
                results[task] = (fallback, 0.0)
            else: # Assuming 'age' is the other task
                results[task] = -1.0

    return results

# ═════════════ CLI TEST (optional) ═════════════
if __name__ == "__main__":
    from pathlib import Path
    sample = Path("sample.jpg")
    if sample.exists():
        out = predict(Image.open(sample).convert("RGB"),
                      tasks={"disease", "variety", "age"})
        print(out)
    else:
        print("Put a test 'sample.jpg' next to inference.py to run CLI demo.")
