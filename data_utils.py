"""
data_utils.py
Shared data loading, model inference, geospatial, and chart-building
helpers used across every tab module (Overview, InputImage, Prediction,
HotspotMapping, ExplainableAI)
"""

# Standard Library
import streamlit as st                         # Web UI
import io, os, re, csv, hashlib, zipfile, sqlite3  # File, data, and DB ops
from datetime import datetime as _dt          # Timestamps
from urllib.parse import unquote, urlparse    # URL parsing

# Data Processing
import numpy as np                             # Numerical ops
import pandas as pd                            # DataFrames

# Visualization
import matplotlib.pyplot as plt               # Static plots
import seaborn as sns                         # Statistical plots
from PIL import Image                         # Image processing

# Geospatial
import folium                                  # Interactive maps
from folium.plugins import MarkerCluster, HeatMap, Fullscreen, MiniMap
from streamlit_folium import st_folium        # Folium + Streamlit

# Deep Learning
import torch                                   # PyTorch core
import torch.nn as nn                         # Neural net layers
from torchvision import models, transforms    # Pre-trained CNNs & transforms

# Set matplotlib style for better visualization
plt.style.use('default')
sns.set_style("whitegrid")

# --------------------------------------------------------------------------- #
# DATA PATHS
# --------------------------------------------------------------------------- #
# Specify the data directory
DATA_DIR = r"C:\Users\Yashreen\Downloads\coral-health-classification\src\components\data"

# Construct paths using os.path.join
CSV_PATH = os.path.join(DATA_DIR, 'coral_dataset_cleaned.csv')
ZIP_PATH = os.path.join(DATA_DIR, 'coral_images_processed.zip')

# --------------------------------------------------------------------------- #
# LABEL MAPS / COLOURS (matches the EDA notebook)
# --------------------------------------------------------------------------- #
HEALTH_MAP = {"1": "Healthy", "2": "Compromised", "3": "Dead"}
STRESSOR_MAP = {"4": "Rubble", "5": "Competition", "6": "Disease",
                 "7": "Predation", "8": "Physical"}

HEALTH_COLORS = {"Healthy": "#27ae60", "Compromised": "#e67e22", "Dead": "#c0392b"}
STRESSOR_COLORS = {"Rubble": "#7f8c8d", "Competition": "#8e44ad",
                    "Disease": "#2980b9", "Predation": "#d35400", "Physical": "#16a085"}

DASH_BG = "#0b2b33"
DASH_TEXT = "#b4e2e9"
DASH_SUBTEXT = "#95c6ce"

# --------------------------------------------------------------------------- #
# MODEL CONFIG + INFERENCE (EfficientNet-B3, 3-class MULTI-LABEL classification)
# --------------------------------------------------------------------------- #
MODEL_PATH = r"C:\Users\Yashreen\Downloads\coral-health-classification\src\components\data\efficientnetb3_tuned_best.pth"

ALL_LABELS = ["Healthy", "Compromised", "Dead"]
HEALTH_LABELS = ALL_LABELS[:3]
STRESSOR_LABELS = []

IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PRED_THRESHOLD = 0.5

RESIZE_MODE = "stretch"  # "stretch" or "centercrop"

HEALTH_RECOMMENDATIONS = {
    "Healthy": "✅ Coral appears healthy. Continue regular monitoring and maintain current protection measures.",
    "Compromised": "⚠️ Coral shows signs of stress. Implement targeted monitoring and consider intervention if condition worsens.",
    "Dead": "💀 Coral mortality detected. Assess surrounding area, identify cause, and plan restoration if feasible."
}
STRESSOR_RECOMMENDATIONS = {}


# --------------------------------------------------------------------------- #
# MODEL ARCHITECTURE — faithful inference-time replica of the training class.
# --------------------------------------------------------------------------- #
class CoralTunedEfficientNetB3(nn.Module):
    def __init__(self, num_classes: int = 3, hidden_dim: int = 512,
                 use_batch_norm: bool = True, dropout_rate: float = 0.3,
                 layerwise_dropout: bool = True):
        super().__init__()
        try:
            weights = models.EfficientNet_B3_Weights.DEFAULT
        except AttributeError:
            weights = "IMAGENET1K_V1"
        self.backbone = models.efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        d1 = dropout_rate * 0.8 if layerwise_dropout else dropout_rate
        d2 = dropout_rate * 0.6 if layerwise_dropout else dropout_rate
        layers = [nn.Dropout(p=d1, inplace=True), nn.Linear(in_features, hidden_dim)]
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SiLU())
        layers += [nn.Dropout(p=d2, inplace=True), nn.Linear(hidden_dim, num_classes)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def _infer_head_config(ckpt_state):
    lin = {k: v for k, v in ckpt_state.items()
           if k.startswith("classifier.") and k.endswith(".weight")
           and hasattr(v, "dim") and v.dim() == 2}
    use_bn = any(k.startswith("classifier.") and k.endswith(".running_mean")
                 for k in ckpt_state)
    if not lin:
        return 512, len(ALL_LABELS), use_bn

    def _idx(k):
        try:
            return int(k.split(".")[1])
        except Exception:
            return 0

    ordered = sorted(lin.items(), key=lambda kv: _idx(kv[0]))
    return int(ordered[0][1].shape[0]), int(ordered[-1][1].shape[0]), use_bn


def _tensor_hash(tensor) -> str:
    return hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:10]


def _load_raw_checkpoint(model_path, device):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    raw = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "net", "model", "classifier_state_dict"):
            if key in checkpoint:
                raw = checkpoint[key]
                break
    if not isinstance(raw, dict):
        raise RuntimeError(
            "torch.load() did not return a state_dict (dict of tensors). "
            "This checkpoint may have been saved with torch.save(module) "
            "rather than torch.save(module.state_dict()).")
    return {k[7:] if k.startswith("module.") else k: v for k, v in raw.items()}


@st.cache_resource(show_spinner="🧠 Loading EfficientNet-B3 model...")
def load_model(model_path=MODEL_PATH, num_classes=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
    ckpt_state = _load_raw_checkpoint(model_path, device)
    hidden_dim, ckpt_classes, use_bn = _infer_head_config(ckpt_state)
    model = CoralTunedEfficientNetB3(num_classes=ckpt_classes, hidden_dim=hidden_dim, use_batch_norm=use_bn)
    total = len(model.state_dict())
    try:
        model.load_state_dict(ckpt_state, strict=True)
        loaded_via = (f"CoralTunedEfficientNetB3 strict load "
                      f"(hidden_dim={hidden_dim}, batch_norm={use_bn}, classes={ckpt_classes})")
        matched = total
    except Exception as strict_err:
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        missing, unexpected = list(missing), list(unexpected)
        matched = total - len(missing)
        clf_missing = [k for k in missing if k.startswith("classifier")]
        bb_missing = [k for k in missing if k.startswith("backbone")]
        if clf_missing or len(bb_missing) > 5:
            raise RuntimeError(
                "Checkpoint does not match the reconstructed CoralTunedEfficientNetB3.\n\n"
                f"strict error: {strict_err}\n\n"
                f"Inferred head: hidden_dim={hidden_dim}, batch_norm={use_bn}, classes={ckpt_classes}.\n"
                f"missing (first 15): {missing[:15]}\nunexpected (first 15): {unexpected[:15]}")
        loaded_via = (f"non-strict load (matched {matched}/{total}; "
                      f"missing={len(missing)}, unexpected={len(unexpected)})")
    model.to(device)
    model.eval()
    last_linear = [m for m in model.classifier if isinstance(m, nn.Linear)][-1]
    fingerprint = _tensor_hash(last_linear.weight)
    load_report = {
        "architecture": (f"CoralTunedEfficientNetB3 (efficientnet_b3 backbone + "
                         f"double-layer head, hidden_dim={hidden_dim}, batch_norm={use_bn})"),
        "prefix": loaded_via, "match_pct": 100.0 * matched / total if total else 0.0,
        "matched": matched, "total": total, "device": str(device),
    }
    print(f"[load_model] {loaded_via}. classifier fingerprint={fingerprint} device={device}")
    return model, device, fingerprint, load_report


def clear_model_cache():
    load_model.clear()


def _get_transform():
    if RESIZE_MODE == "centercrop":
        return transforms.Compose([
            transforms.Resize(320), transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def bytes_to_pil_image(file_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(file_bytes)).convert("RGB")


def image_bytes_fingerprint(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# UNDERWATER IMAGE ENHANCEMENT (same recipe as the preprocessing notebook)
# Applied to every uploaded image BEFORE prediction so inference sees the same
# --------------------------------------------------------------------------- #
try:
    import cv2 as _cv2
except Exception:
    _cv2 = None


def _gray_world_white_balance(img_rgb):
    """Neutralise the blue/green underwater colour cast (gray-world assumption)."""
    img_f = img_rgb.astype(np.float32)
    means = img_f.mean(axis=(0, 1))
    scale = means.mean() / (means + 1e-6)
    if scale.max() / (scale.min() + 1e-6) < 1.2:      # already balanced -> skip
        return img_rgb.astype(np.uint8)
    return np.clip(img_f * scale, 0, 255).astype(np.uint8)

def _clahe_lab(img_rgb, tile_grid=(8, 8)):
    """Local contrast enhancement on the L (lightness) channel via CLAHE."""
    lab = _cv2.cvtColor(_cv2.cvtColor(img_rgb, _cv2.COLOR_RGB2BGR), _cv2.COLOR_BGR2LAB)
    l, a, b = _cv2.split(lab)
    mean_brightness = l.mean()
    clip = 4.0 if mean_brightness < 60 else 3.0 if mean_brightness < 100 else 2.0
    l_eq = _cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid).apply(l)
    merged = _cv2.merge([l_eq, a, b])
    return _cv2.cvtColor(_cv2.cvtColor(merged, _cv2.COLOR_LAB2BGR), _cv2.COLOR_BGR2RGB)

def _enhance_underwater(img_rgb, depth=None):
    """White balance + CLAHE, blended with the original by a depth-aware weight."""
    enhanced = _clahe_lab(_gray_world_white_balance(img_rgb))
    alpha = 0.85 if depth is None else 0.65 if depth <= 3 else 0.80 if depth <= 8 else 0.95
    blended = _cv2.addWeighted(enhanced.astype(np.float32), alpha,
                               img_rgb.astype(np.float32), 1.0 - alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

def _normalise_image(img_rgb):
    """Per-image min-max stretch to the full 0-255 range."""
    img_f = img_rgb.astype(np.float32)
    lo, hi = img_f.min(), img_f.max()
    if hi - lo < 1e-6:
        return img_rgb.astype(np.uint8)
    return ((img_f - lo) / (hi - lo) * 255.0).astype(np.uint8)


def enhance_pil_image(pil_image, depth=None):
    """
    Enhance (underwater white balance + CLAHE) then normalise a PIL image and
    return a PIL image. If OpenCV is unavailable, returns the original unchanged.
    """
    if _cv2 is None:
        return pil_image.convert("RGB")
    arr = np.array(pil_image.convert("RGB"))
    arr = _enhance_underwater(arr, depth=depth)
    arr = _normalise_image(arr)
    return Image.fromarray(arr)


def enhance_available() -> bool:
    """True if OpenCV is installed so enhancement can run."""
    return _cv2 is not None


def enhance_bytes(file_bytes, depth=None):
    """Enhance raw image bytes -> return JPEG bytes of the processed image."""
    pil = enhance_pil_image(bytes_to_pil_image(file_bytes), depth=depth)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def predict_image(pil_image, debug: bool = False):
    model, device, _fingerprint, _load_report = load_model()
    tfm = _get_transform()
    x = tfm(pil_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    if debug:
        print(f"[predict_image] mean={x.mean().item():.4f} std={x.std().item():.4f}")
        print(f"[predict_image] logits={logits.cpu().numpy().tolist()}")
        print(f"[predict_image] sigmoid probs={probs.tolist()}")
    return {label: float(p) for label, p in zip(ALL_LABELS, probs)}


def predicted_label_vector(probs: dict, threshold: float = PRED_THRESHOLD):
    top_label = max(probs, key=probs.get)
    out = []
    for label in ALL_LABELS:
        p = float(probs.get(label, 0.0))
        out.append({"label": label, "prob": p, "bit": 1 if p >= threshold else 0,
                    "is_top": label == top_label})
    return out


# --------------------------------------------------------------------------- #
# GRAD-CAM
# --------------------------------------------------------------------------- #
def _gradcam_target_layer(model):
    return model.backbone.features[-1]


def compute_gradcam(pil_image, target_label=None, alpha: float = 0.45):
    model, device, _fingerprint, _load_report = load_model()
    tfm = _get_transform()
    orig = pil_image.convert("RGB")
    x = tfm(orig).unsqueeze(0).to(device)
    activations, gradients = {}, {}
    target_layer = _gradcam_target_layer(model)

    def _fwd_hook(_m, _i, out):
        activations["value"] = out

    def _bwd_hook(_m, _gi, grad_out):
        gradients["value"] = grad_out[0]

    h_fwd = target_layer.register_forward_hook(_fwd_hook)
    try:
        h_bwd = target_layer.register_full_backward_hook(_bwd_hook)
    except AttributeError:
        h_bwd = target_layer.register_backward_hook(_bwd_hook)
    try:
        with torch.enable_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)
            if target_label is not None and target_label in ALL_LABELS:
                class_idx = ALL_LABELS.index(target_label)
            else:
                class_idx = int(torch.argmax(logits, dim=1).item())
            model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()
        acts = activations["value"].detach()[0]
        grads = gradients["value"].detach()[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0)).cpu().numpy().astype(np.float32)
    finally:
        h_fwd.remove()
        h_bwd.remove()
    cam -= cam.min()
    if cam.max() > 1e-8:
        cam /= cam.max()
    W, H = orig.size
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((W, H), resample=Image.BILINEAR)
    cam_arr = np.asarray(cam_img).astype(np.float32) / 255.0
    heat = (plt.cm.jet(cam_arr)[:, :, :3] * 255).astype(np.uint8)
    orig_arr = np.asarray(orig).astype(np.float32)
    overlay = np.uint8((1 - alpha) * orig_arr + alpha * heat.astype(np.float32))
    return {"overlay": Image.fromarray(overlay), "heatmap": Image.fromarray(heat),
            "original": orig, "class_label": ALL_LABELS[class_idx],
            "class_conf": float(probs[0, class_idx].item())}


# --------------------------------------------------------------------------- #
# DATA LOADING
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    for lid, name in HEALTH_MAP.items():
        df[f"has_{name.lower()}"] = df["coral_health_labels"].apply(
            lambda v, lid=lid: 1 if pd.notna(v) and lid in str(v).split(",") else 0)
    for lid, name in STRESSOR_MAP.items():
        df[f"has_{name.lower()}"] = df["stressor_labels"].apply(
            lambda v, lid=lid: 1 if pd.notna(v) and lid in str(v).split(",") else 0)

    def primary_health(val):
        if pd.isna(val):
            return "Unknown"
        ids = str(val).split(",")
        if "3" in ids:
            return "Dead"
        if "2" in ids:
            return "Compromised"
        if "1" in ids:
            return "Healthy"
        return "Unknown"

    df["primary_health"] = df["coral_health_labels"].apply(primary_health)
    if "image_url" in df.columns:
        df["season"] = df["image_url"].apply(lambda u: "Wet" if "coral_images_wet" in str(u) else "Dry")
    df["depth"] = pd.to_numeric(df.get("depth"), errors="coerce")
    df["temp"] = pd.to_numeric(df.get("temp"), errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_image_zip(zip_path):
    zf = zipfile.ZipFile(zip_path, "r")
    index = {}
    for name in zf.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            index[os.path.basename(name)] = name
    return zf, index


def get_image_from_zip(zf, index, image_url):
    if zf is None or not isinstance(image_url, str):
        return None
    basename = os.path.basename(unquote(urlparse(image_url).path)) or os.path.basename(image_url)
    entry = index.get(basename)
    if entry is None:
        stem = os.path.splitext(basename)[0]
        matches = [v for k, v in index.items() if stem in k]
        entry = matches[0] if matches else None
    if entry is None:
        return None
    try:
        with zf.open(entry) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# GEOSPATIAL HELPERS
# --------------------------------------------------------------------------- #
HEALTH_MARKER_COLORS = {"Healthy": "#10B981", "Compromised": "#F59E0B", "Dead": "#EF4444"}


def dms_to_decimal(dms):
    if pd.isna(dms):
        return None
    s = str(dms).strip()
    m = re.match(r"(\d+)\D+(\d+)\D+(\d+(?:\.\d+)?)\D*([NSEWnsew])", s)
    if not m:
        return None
    deg, minute, sec, direction = m.groups()
    dec = float(deg) + float(minute) / 60 + float(sec) / 3600
    if direction.upper() in ("S", "W"):
        dec = -dec
    return dec


@st.cache_data(show_spinner=False)
def build_site_geo(df):
    d = df.copy()
    d["lat"] = d["lat_start"].apply(dms_to_decimal)
    d["lon"] = d["lng_start"].apply(dms_to_decimal)
    d["survey_date_parsed"] = pd.to_datetime(
        d["survey_date"].astype(str), format="%Y%m%d", errors="coerce") if "survey_date" in d.columns else pd.NaT
    rows = []
    group_cols = ["site", "site_fullname"] if "site_fullname" in d.columns else ["site"]
    for keys, g in d.groupby(group_cols):
        site = keys[0] if isinstance(keys, tuple) else keys
        fullname = keys[1] if isinstance(keys, tuple) and len(keys) > 1 else site
        total = len(g)
        vc = g["primary_health"].value_counts()
        n_healthy = vc.get("Healthy", 0); n_comp = vc.get("Compromised", 0); n_dead = vc.get("Dead", 0)
        pct_h = 100 * n_healthy / total if total else 0
        pct_c = 100 * n_comp / total if total else 0
        pct_d = 100 * n_dead / total if total else 0
        score = (pct_h * 1.0 + pct_c * 0.5) / 100
        dominant = max([("Healthy", pct_h), ("Compromised", pct_c), ("Dead", pct_d)], key=lambda t: t[1])[0]
        rows.append(dict(
            site=site, site_fullname=fullname,
            lat=g["lat"].dropna().iloc[0] if g["lat"].notna().any() else None,
            lon=g["lon"].dropna().iloc[0] if g["lon"].notna().any() else None,
            total=total, n_healthy=n_healthy, n_compromised=n_comp, n_dead=n_dead,
            pct_healthy=pct_h, pct_compromised=pct_c, pct_dead=pct_d,
            health_score=score, dominant_health=dominant,
            mean_depth=g["depth"].mean(), min_depth=g["depth"].min(), max_depth=g["depth"].max(),
            mean_temp=g["temp"].mean(), min_temp=g["temp"].min(), max_temp=g["temp"].max(),
            last_survey=g["survey_date_parsed"].max() if "survey_date_parsed" in g.columns else None))
    site_geo = pd.DataFrame(rows).sort_values("health_score", ascending=False).reset_index(drop=True)
    site_geo["rank"] = np.arange(1, len(site_geo) + 1)
    return site_geo


# --------------------------------------------------------------------------- #
# CHART BUILDERS
# --------------------------------------------------------------------------- #
def style_axes(ax):
    ax.set_facecolor("#f8f9fa")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, alpha=0.3)


def chart_class_distribution(df):
    health_names = list(HEALTH_MAP.values())
    counts = {n: df[f"has_{n.lower()}"].sum() for n in health_names}
    total = len(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Coral Reef Health Label Class Distribution", fontsize=14, fontweight="bold", y=1.02)
    bars = axes[0].bar(counts.keys(), counts.values(), color=[HEALTH_COLORS[k] for k in counts],
                       width=0.5, edgecolor="white", linewidth=1.2)
    for bar, (name, val) in zip(bars, counts.items()):
        pct = 100 * val / total if total else 0
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + total * 0.01, f"{val:,}\n({pct:.1f}%)",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Number of Images", fontsize=11)
    axes[0].set_title("Absolute Count", fontsize=11, fontweight="bold")
    style_axes(axes[0])
    prim = df[df["primary_health"] != "Unknown"]["primary_health"].value_counts()
    if not prim.empty:
        axes[1].pie(prim.values, labels=prim.index, colors=[HEALTH_COLORS.get(k, "#aaa") for k in prim.index],
                    autopct="%1.1f%%", startangle=140, pctdistance=0.78,
                    wedgeprops=dict(edgecolor="white", linewidth=2),
                    textprops={'fontsize': 10, 'fontweight': 'bold'})
    axes[1].set_title("Primary Health Condition\n(worst-case label per image)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_stressor_distribution(df):
    total = len(df)
    stressor_names = list(STRESSOR_MAP.values())
    counts = {n: df[f"has_{n.lower()}"].sum() for n in stressor_names}
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    no_stressor = df["stressor_labels"].isna().sum() if "stressor_labels" in df.columns else 0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Stressor Label Distribution", fontsize=14, fontweight="bold", y=1.02)
    colors = [STRESSOR_COLORS[k] for k in counts]
    bars = axes[0].barh(list(counts.keys()), list(counts.values()), color=colors, height=0.5,
                        edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values()):
        pct = 100 * val / total if total else 0
        axes[0].text(val + total * 0.005, bar.get_y() + bar.get_height() / 2, f"  {val:,} ({pct:.1f}%)",
                     va="center", fontsize=9, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of Images", fontsize=11)
    style_axes(axes[0])
    pie_vals = list(counts.values()) + [no_stressor]
    pie_labels = list(counts.keys()) + ["No Stressor"]
    pie_colors = colors + ["#dfe6e9"]
    axes[1].pie(pie_vals, labels=pie_labels, colors=pie_colors, autopct="%1.1f%%", startangle=140,
                pctdistance=0.78, wedgeprops=dict(edgecolor="white", linewidth=1.5),
                textprops={'fontsize': 9, 'fontweight': 'bold'})
    axes[1].set_title("Stressor Proportion\n(including no-stressor images)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    return fig


def chart_class_imbalance(df):
    total = len(df)
    names = list(HEALTH_MAP.values()) + list(STRESSOR_MAP.values())
    counts = [df[f"has_{n.lower()}"].sum() for n in names]
    colors = [HEALTH_COLORS[n] for n in HEALTH_MAP.values()] + [STRESSOR_COLORS[n] for n in STRESSOR_MAP.values()]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Class Imbalance Overview (All Labels)", fontsize=14, fontweight="bold", y=1.02)
    x = range(len(names))
    bars = ax.bar(x, counts, color=colors, width=0.6, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts):
        pct = 100 * val / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, val + total * 0.005, f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel("Number of Images", fontsize=11)
    if names:
        ax.axhline(total / len(names), color="grey", linestyle="--", linewidth=1.2, label="Average if balanced")
        ax.legend(fontsize=9)
    ax.axvline(2.5, color="#bdc3c7", linestyle="-", linewidth=1.5)
    style_axes(ax)
    plt.tight_layout()
    return fig


def chart_cooccurrence(df):
    cols = [f"has_{n.lower()}" for n in list(HEALTH_MAP.values()) + list(STRESSOR_MAP.values())]
    names = list(HEALTH_MAP.values()) + list(STRESSOR_MAP.values())
    mat = df[cols].values.astype(int)
    co = mat.T @ mat
    mask = np.eye(len(cols), dtype=bool)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Label Co-occurrence Heatmap\n(diagonal masked)", fontsize=14, fontweight="bold", y=0.98)
    sns.heatmap(co, mask=mask, annot=True, fmt="d", xticklabels=names, yticklabels=names,
                cmap="YlOrRd", linewidths=0.5, linecolor="white",
                annot_kws={"size": 9, "fontweight": "bold"}, ax=ax, cbar_kws={"label": "Co-occurrence count"})
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        name = tick.get_text()
        tick.set_color(HEALTH_COLORS.get(name, STRESSOR_COLORS.get(name, "black")))
        tick.set_fontweight("bold")
        tick.set_fontsize(10)
    plt.tight_layout()
    return fig


def chart_health_per_site(df):
    if "site" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No site data available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    health_names = list(HEALTH_MAP.values())
    site_df = df.groupby("site")[[f"has_{n.lower()}" for n in health_names]].sum()
    site_df.columns = health_names
    site_pct = site_df.div(site_df.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Coral Health Distribution Per Survey Site", fontsize=14, fontweight="bold", y=1.02)
    sites = site_df.index.tolist()
    x = np.arange(len(sites))
    for ax, data, ylabel in zip(axes, [site_df, site_pct], ["Number of Images", "Percentage (%)"]):
        bottom = np.zeros(len(sites))
        for name in health_names:
            vals = data[name].fillna(0)
            bars = ax.bar(x, vals, bottom=bottom, color=HEALTH_COLORS[name], label=name,
                          edgecolor="white", linewidth=0.8)
            for i, (bar, val) in enumerate(zip(bars, vals)):
                if val > (3 if ylabel == "Percentage (%)" else max(vals.max() * 0.02, 1)):
                    ax.text(bar.get_x() + bar.get_width() / 2, bottom[i] + val / 2,
                            f"{val:.0f}{'%' if ylabel == 'Percentage (%)' else ''}",
                            ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            bottom += vals.values
        ax.set_xticks(x)
        ax.set_xticklabels(sites, fontsize=9, rotation=45, ha='right')
        ax.set_xlabel("Survey Site", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(title="Health", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        style_axes(ax)
    plt.tight_layout()
    return fig


def chart_site_scorecard(df):
    if "site" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No site data available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig, pd.DataFrame()
    site_summary = df.groupby("site").agg(
        total=("primary_health", "count"), n_healthy=("has_healthy", "sum"),
        n_compromised=("has_compromised", "sum"), n_dead=("has_dead", "sum"),
        mean_depth=("depth", "mean"), mean_temp=("temp", "mean")).reset_index()
    site_summary["pct_healthy"] = site_summary["n_healthy"] / site_summary["total"] * 100
    site_summary["pct_compromised"] = site_summary["n_compromised"] / site_summary["total"] * 100
    site_summary["pct_dead"] = site_summary["n_dead"] / site_summary["total"] * 100
    site_summary["health_score"] = (site_summary["pct_healthy"] * 1.0 + site_summary["pct_compromised"] * 0.5) / 100
    site_summary = site_summary.sort_values("health_score", ascending=False).reset_index(drop=True)
    site_summary["rank"] = site_summary.index + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Site Health Scorecard\n(score = %Healthy×1.0 + %Compromised×0.5)", fontsize=14, fontweight="bold", y=1.02)
    score_colors = ["#27ae60" if s >= 0.6 else "#e67e22" if s >= 0.4 else "#c0392b" for s in site_summary["health_score"]]
    bars = axes[0].barh(site_summary["site"][::-1], site_summary["health_score"][::-1],
                        color=score_colors[::-1], height=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, site_summary["health_score"][::-1]):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height() / 2, f" {val:.2f}", va="center", fontsize=10, fontweight="bold")
    axes[0].axvline(0.5, color="#bdc3c7", linestyle="--", linewidth=1.2, label="0.5 threshold")
    axes[0].set_xlim(0, 1.15)
    axes[0].set_title("Overall Health Score (ranked)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    style_axes(axes[0])
    x = np.arange(len(site_summary))
    bottom = np.zeros(len(site_summary))
    for name, col in zip(list(HEALTH_MAP.values()), ["pct_healthy", "pct_compromised", "pct_dead"]):
        bars = axes[1].bar(x, site_summary[col], bottom=bottom, color=HEALTH_COLORS[name],
                           label=name, width=0.5, edgecolor="white", linewidth=1.2)
        for i, (bar, val) in enumerate(zip(bars, site_summary[col])):
            if val > 5:
                axes[1].text(bar.get_x() + bar.get_width() / 2, bottom[i] + val / 2, f"{val:.0f}%",
                             ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += site_summary[col].values
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(site_summary["site"], fontsize=9, rotation=45, ha='right')
    axes[1].set_title("Health Breakdown % (ranked)", fontsize=11, fontweight="bold")
    axes[1].legend(title="Health", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    style_axes(axes[1])
    plt.tight_layout()
    return fig, site_summary


def chart_depth_vs_health(df):
    if "depth" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No depth data available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    d = df[df["primary_health"] != "Unknown"].dropna(subset=["depth"]).copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient depth data", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    order = ["Healthy", "Compromised", "Dead"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Water Depth by Coral Health", fontsize=14, fontweight="bold", y=1.02)
    sns.boxplot(data=d, x="primary_health", y="depth", order=order, palette=HEALTH_COLORS,
                width=0.45, linewidth=1.5, ax=axes[0], medianprops=dict(color="white", linewidth=2.5))
    axes[0].set_title("Depth Distribution by Health", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Health Condition", fontsize=11)
    axes[0].set_ylabel("Depth (m)", fontsize=11)
    style_axes(axes[0])
    site_depth = df.groupby("site").agg(
        mean_depth=("depth", "mean"),
        dominant=("primary_health", lambda x: x.value_counts().idxmax() if len(x) else "Unknown")).reset_index()
    bars = axes[1].bar(site_depth["site"], site_depth["mean_depth"],
                       color=[HEALTH_COLORS.get(dd, "#aaa") for dd in site_depth["dominant"]],
                       width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, site_depth["mean_depth"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.2, f"{val:.1f}m",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axes[1].set_title("Mean Depth per Site\n(colour = dominant health)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Survey Site", fontsize=11)
    axes[1].set_ylabel("Mean Depth (m)", fontsize=11)
    axes[1].set_xticklabels(site_depth["site"], rotation=45, ha='right')
    style_axes(axes[1])
    plt.tight_layout()
    return fig


def chart_temp_vs_health(df):
    if "temp" not in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No temperature data available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    d = df[df["primary_health"] != "Unknown"].dropna(subset=["temp"]).copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient temperature data", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return fig
    order = ["Healthy", "Compromised", "Dead"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Water Temperature by Coral Health", fontsize=14, fontweight="bold", y=1.02)
    for name in order:
        subset = d[d["primary_health"] == name]["temp"]
        if len(subset) > 1:
            subset.plot.kde(ax=axes[0], color=HEALTH_COLORS[name], linewidth=2.5, label=f"{name} (n={len(subset):,})")
    axes[0].set_title("Temperature Distribution (KDE)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Water Temperature (°C)", fontsize=11)
    axes[0].set_ylabel("Density", fontsize=11)
    axes[0].legend(fontsize=9)
    style_axes(axes[0])
    site_scatter = df.groupby("site").agg(
        mean_temp=("temp", "mean"), pct_dead=("has_dead", "mean"), total=("primary_health", "count")).reset_index()
    site_scatter["pct_dead"] *= 100
    sc = axes[1].scatter(site_scatter["mean_temp"], site_scatter["pct_dead"],
                         s=site_scatter["total"] / site_scatter["total"].max() * 400 + 80,
                         c=site_scatter["pct_dead"], cmap="Reds", edgecolors="white", linewidth=1.5, vmin=0, vmax=100)
    for _, row in site_scatter.iterrows():
        axes[1].annotate(row["site"], xy=(row["mean_temp"], row["pct_dead"]), xytext=(4, 4),
                         textcoords="offset points", fontsize=9, fontweight="bold")
    plt.colorbar(sc, ax=axes[1], label="% Dead coral", fraction=0.046, pad=0.04)
    axes[1].set_title("Temp vs % Dead per Site\n(bubble size = image count)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Mean Water Temperature (°C)", fontsize=11)
    axes[1].set_ylabel("% Images with Dead Coral", fontsize=11)
    style_axes(axes[1])
    plt.tight_layout()
    return fig


# =========================================================================== #
# LIVE SQL DATABASE + HUMAN-ANNOTATION SAVE  (four linked tables)
# ---------------------------------------------------------------------------
# On save, one reviewed coral patch is written in real time to:
#   * SQLite (coral_live.db) with FOUR related tables:
#         labelset (reference: id -> Healthy/.../Physical)
#         surveys        1 --< coral_dataset  (one survey has many patches)
#         coral_dataset  1 --< annotations    (one patch has many annotations)
#         surveys        1 --< annotations
#   * coral_dataset_cleaned.csv  (one denormalised row per patch)
#   * surveys_metadata.csv       (one row per survey; added only if new)
#   * annotations.csv            (one row per human/AI annotation event)
#   * the image file under saved_patches/<folder>/<file>.
# =========================================================================== 
METADATA_CSV_PATH = os.path.join(DATA_DIR, "surveys_metadata.csv")
ANNOTATIONS_CSV_PATH = os.path.join(DATA_DIR, "annotations.csv")
DB_PATH = os.path.join(DATA_DIR, "coral_live.db")
SAVED_IMAGES_DIR = os.path.join(DATA_DIR, "saved_patches")

# label name -> dataset id (reverse of HEALTH_MAP / STRESSOR_MAP)
HEALTH_LABEL_ID = {v: k for k, v in HEALTH_MAP.items()}
STRESSOR_LABEL_ID = {v: k for k, v in STRESSOR_MAP.items()}

# Column order of surveys_metadata.csv (survey-level fields)
SURVEY_COLUMNS = ["surveyid", "transectid", "survey_date", "site", "site_fullname",
                  "folder_name", "lat_start", "lng_start", "camera", "depth", "temp"]

# Column order of coral_dataset_cleaned.csv (denormalised, one row per patch)
PATCH_COLUMNS = ["image_filename", "image_url", "patchid", "site_from_filename",
                 "survey_date_from_filename", "surveyid", "transectid", "survey_date",
                 "site", "site_fullname", "folder_name", "lat_start", "lng_start",
                 "camera", "depth", "temp", "coral_health_labels", "stressor_labels"]

# annotations.csv is kept minimal: exactly two columns matching the ground-truth
# file (one row per patch). All rich annotation metadata lives in the SQLite
# `annotations` table instead.
#   patchid : uploaded image name (filename stem), e.g. CBK_0039_00_20240126_0038_35
#   label   : all applicable label ids — health (1=Healthy, 2=Compromised,
#             3=Dead) plus human-added stressors (4-8), e.g. "2,6"
ANNOTATIONS_CSV_COLUMNS = ["patchid", "label"]

# Full column set retained for the SQLite `annotations` table / internal use.
ANNOTATION_COLUMNS = ["annotation_id", "image_filename", "patchid", "label", "surveyid",
                      "survey_date", "site", "source", "ai_health_labels",
                      "human_health_labels", "coral_health_labels", "stressor_labels",
                      "annotated_by", "created_at"]

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS labelset (
    label_id   TEXT PRIMARY KEY,          -- '1'..'8'
    label_type TEXT,                       -- 'health' | 'stressor'
    label_name TEXT                        -- 'Healthy', 'Disease', ...
);
CREATE TABLE IF NOT EXISTS surveys (
    surveyid      TEXT PRIMARY KEY,
    transectid    TEXT,
    survey_date   TEXT,
    site          TEXT,
    site_fullname TEXT,
    folder_name   TEXT,
    lat_start     TEXT,
    lng_start     TEXT,
    camera        TEXT,
    depth         TEXT,
    temp          TEXT
);
CREATE TABLE IF NOT EXISTS coral_dataset (
    image_filename            TEXT PRIMARY KEY,
    image_url                 TEXT,
    patchid                   TEXT,
    site_from_filename        TEXT,
    survey_date_from_filename TEXT,
    surveyid                  TEXT,
    survey_date               TEXT,
    site                      TEXT,
    folder_name               TEXT,
    coral_health_labels       TEXT,
    stressor_labels           TEXT,
    depth                     REAL,
    temp                      REAL,
    created_at                TEXT,
    FOREIGN KEY (surveyid) REFERENCES surveys(surveyid)
);
CREATE TABLE IF NOT EXISTS annotations (
    annotation_id       TEXT PRIMARY KEY,
    image_filename      TEXT,
    patchid             TEXT,
    label               TEXT,              -- combined ids (health + stressor), e.g. '2,6'
    surveyid            TEXT,
    survey_date         TEXT,
    site                TEXT,
    source              TEXT,              -- 'ai_accepted' | 'human_corrected'
    ai_health_labels    TEXT,
    human_health_labels TEXT,
    coral_health_labels TEXT,
    stressor_labels     TEXT,
    annotated_by        TEXT,
    created_at          TEXT,
    FOREIGN KEY (image_filename) REFERENCES coral_dataset(image_filename),
    FOREIGN KEY (surveyid)       REFERENCES surveys(surveyid)
);
CREATE INDEX IF NOT EXISTS idx_cd_surveyid    ON coral_dataset(surveyid);
CREATE INDEX IF NOT EXISTS idx_cd_survey_date ON coral_dataset(survey_date);
CREATE INDEX IF NOT EXISTS idx_cd_folder      ON coral_dataset(folder_name);
CREATE INDEX IF NOT EXISTS idx_anno_image     ON annotations(image_filename);
CREATE INDEX IF NOT EXISTS idx_anno_surveyid  ON annotations(surveyid);
"""


# Columns each table must have. If an OLDER coral_live.db is missing any of
# them (because it was created before the schema changed), init_db() adds them
# with ALTER TABLE — CREATE TABLE IF NOT EXISTS never alters an existing table,
# which is why "table annotations has no column named label" kept appearing.
_REQUIRED_COLUMNS = {
    "annotations": {
        "label": "TEXT", "surveyid": "TEXT", "survey_date": "TEXT", "site": "TEXT",
        "source": "TEXT", "ai_health_labels": "TEXT", "human_health_labels": "TEXT",
        "coral_health_labels": "TEXT", "stressor_labels": "TEXT",
        "annotated_by": "TEXT", "created_at": "TEXT",
    },
    "coral_dataset": {
        "coral_health_labels": "TEXT", "stressor_labels": "TEXT",
        "depth": "REAL", "temp": "REAL", "created_at": "TEXT",
    },
}


def _migrate_schema(con):
    """Add any missing columns to existing tables (idempotent)."""
    for table, cols in _REQUIRED_COLUMNS.items():
        try:
            existing = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        except Exception:
            continue
        if not existing:          # table doesn't exist yet; SCHEMA_SQL creates it
            continue
        for col, coltype in cols.items():
            if col not in existing:
                try:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                except Exception:
                    pass


def init_db():
    """Create the SQLite DB + tables if needed, migrate older DBs, seed the labelset.
    Safe to call often."""
    if DATA_DIR:
        os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(SCHEMA_SQL)     # create anything missing
        _migrate_schema(con)              # patch older tables that predate new columns
        for lid, name in HEALTH_MAP.items():
            con.execute("INSERT OR IGNORE INTO labelset VALUES (?,?,?)", (lid, "health", name))
        for lid, name in STRESSOR_MAP.items():
            con.execute("INSERT OR IGNORE INTO labelset VALUES (?,?,?)", (lid, "stressor", name))
        con.commit()
    finally:
        con.close()
    return DB_PATH


def _num(value):
    """Extract a float from strings like '0.3m', '31.3℃', '18m' -> 0.3 / 31.3 / 18."""
    if value is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(m.group()) if m else None


def labels_to_ids(names, mapping):
    """['Compromised','Dead'] + HEALTH_LABEL_ID -> '2,3' (sorted, comma-joined)."""
    ids = sorted((mapping[n] for n in names if n in mapping), key=lambda s: int(s))
    return ",".join(ids)


def load_surveys_metadata():
    """Read surveys_metadata.csv (for autofill). Returns a DataFrame (possibly empty)."""
    if os.path.exists(METADATA_CSV_PATH):
        try:
            return pd.read_csv(METADATA_CSV_PATH, dtype=str).fillna("")
        except Exception:
            pass
    return pd.DataFrame(columns=SURVEY_COLUMNS)


# ---- Autofill helpers used by the Prediction "Human Review" flow ---------- #
def get_site_options():
    """List of (site_code, site_fullname) for existing sites, de-duplicated."""
    md = load_surveys_metadata()
    if md.empty or "site" not in md.columns:
        return []
    seen = {}
    for _, r in md.iterrows():
        s = str(r.get("site", "")).strip()
        if s and s not in seen:
            seen[s] = str(r.get("site_fullname", "")).strip()
    return [(s, fn) for s, fn in seen.items()]


def get_site_defaults(site):
    """Autofill dict (site, site_fullname, lat_start, lng_start, camera, depth, temp) for a site."""
    md = load_surveys_metadata()
    if md.empty or "site" not in md.columns:
        return {}
    sub = md[md["site"].astype(str) == str(site)]
    if sub.empty:
        return {}
    row = sub.iloc[-1]
    return {k: str(row.get(k, "")) for k in
            ["site", "site_fullname", "lat_start", "lng_start", "camera", "depth", "temp"]}


def get_surveys_for_site(site):
    """All surveys_metadata rows for one site (for the 'existing survey date' picker)."""
    md = load_surveys_metadata()
    if md.empty or "site" not in md.columns:
        return md
    return md[md["site"].astype(str) == str(site)]


# Patch filenames follow: SITE_SURVEY_TRANSECT_YYYYMMDD_FRAME_PATCH
#   e.g. CBK_0039_00_20240126_0038_35(.jpg)  ->
#        site=CBK, surveyid=0039, transectid=00, survey_date=20240126,
#        frame=0038, patch_no=35
_PATCH_NAME_RE = re.compile(r"^([A-Za-z]+)_(\d+)_(\d+)_(\d{8})_(\d+)_(\d+)$")


def parse_patch_filename(name):
    """Parse a coral patch image filename into its metadata parts.

    Returns a dict. `patchid` is always the filename stem (no extension), so the
    caller can key the annotation on it. If the name matches the survey naming
    convention, site / surveyid / transectid / survey_date / frame / patch_no are
    also filled in for auto-fill. Non-matching names just get {'patchid': stem}.
    """
    if not name:
        return {}
    stem = os.path.splitext(os.path.basename(str(name)))[0]
    m = _PATCH_NAME_RE.match(stem)
    if not m:
        return {"patchid": stem}
    site, surveyid, transectid, sdate, frame, patch_no = m.groups()
    return {"patchid": stem, "site": site.upper(), "surveyid": surveyid,
            "transectid": transectid, "survey_date": sdate,
            "frame": frame, "patch_no": patch_no}


def get_survey_defaults_from_filename(name):
    """Auto-fill dict for the annotation form, derived from an uploaded filename.

    Combines the parts parsed from the filename with any matching row already in
    surveys_metadata.csv (looked up by surveyid, then by site) so the researcher
    sees site, survey date, lat/lng, depth, temp, camera pre-filled.
    """
    parsed = parse_patch_filename(name)
    if not parsed:
        return {}
    out = {"patchid": parsed.get("patchid", "")}
    site = parsed.get("site", "")
    surveyid = parsed.get("surveyid", "")
    md = load_surveys_metadata()
    row = {}
    if not md.empty:
        if surveyid and "surveyid" in md.columns:
            sub = md[md["surveyid"].astype(str) == str(surveyid)]
            if not sub.empty:
                row = sub.iloc[-1].to_dict()
        if not row and site and "site" in md.columns:
            sub = md[md["site"].astype(str) == str(site)]
            if not sub.empty:
                row = sub.iloc[-1].to_dict()
    # site-level fallbacks first, then the specific metadata row, then filename
    out.update(get_site_defaults(site))
    out.update({k: str(v) for k, v in row.items() if str(v).strip()})
    if site:
        out["site"] = site
    for k in ("surveyid", "transectid", "survey_date"):
        if parsed.get(k):
            out[k] = parsed[k]
    return out


def next_surveyid():
    """Next zero-padded survey id, based on the numeric max already in metadata."""
    md = load_surveys_metadata()
    mx = 0
    if not md.empty and "surveyid" in md.columns:
        for v in md["surveyid"]:
            try:
                mx = max(mx, int(str(v)))
            except Exception:
                pass
    return f"{mx + 1:04d}"


def _existing_surveyids():
    if os.path.exists(METADATA_CSV_PATH):
        try:
            return set(pd.read_csv(METADATA_CSV_PATH, dtype=str)["surveyid"].astype(str))
        except Exception:
            return set()
    return set()


def _count(con, table):
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] or 0
    except Exception:
        return 0


def _next_patchid(con, folder_name):
    n = _count_where(con, "coral_dataset", "folder_name", folder_name)
    return f"{n:04d}"


def _count_where(con, table, col, val):
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = ?", (val,)).fetchone()[0] or 0
    except Exception:
        return 0


def _next_annotation_id(con):
    n = _count(con, "annotations")
    return f"A{n + 1:06d}"


def _append_csv_row(path, default_header, row_dict):
    """Append one row to a CSV efficiently; use the file's existing header order if present."""
    file_exists = os.path.exists(path)
    header = default_header
    if file_exists:
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                first = f.readline().strip("\n")
            if first:
                header = next(csv.reader([first]))
        except Exception:
            header = default_header
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow([row_dict.get(col, "") for col in header])


def save_patch_record(health_labels, stressor_labels, meta, image_bytes,
                      source_name=None, ai_health_labels=None,
                      source=None, annotated_by="researcher"):
    """
    Save one human-reviewed coral patch to the live database + all CSVs, and
    write the image to disk. Returns a summary dict.

    health_labels / stressor_labels : final label NAMES (e.g. ['Compromised'])
    ai_health_labels : what the AI predicted (NAMES) — recorded on the annotation
    source : 'ai_accepted' | 'human_corrected' (auto-derived if None)
    meta : dict with SURVEY_COLUMNS keys; survey_date should be 'YYYYMMDD'.
    """
    init_db()
    survey_date = str(meta.get("survey_date", "")).strip()
    site = str(meta.get("site", "")).strip()
    folder_name = str(meta.get("folder_name") or "").strip() or (
        f"{survey_date}_{site}" if survey_date and site else (survey_date or site or "unsorted"))

    con = sqlite3.connect(DB_PATH)
    try:
        # patchid comes from the uploaded image's filename (its stem), e.g.
        # 'CBK_0039_00_20240126_0038_35.jpg' -> 'CBK_0039_00_20240126_0038_35'.
        # Fall back to a generated sequential id only if no filename is given.
        parsed = parse_patch_filename(source_name) if source_name else {}
        patchid = str(meta.get("patchid") or parsed.get("patchid") or "").strip()
        if patchid:
            image_filename = os.path.basename(str(source_name)) if source_name else f"{patchid}.jpg"
            if not os.path.splitext(image_filename)[1]:
                image_filename += ".jpg"
        else:
            patchid = _next_patchid(con, folder_name)
            image_filename = f"{folder_name}_{patchid}.jpg"

        # 1) write the image file
        fdir = os.path.join(SAVED_IMAGES_DIR, folder_name)
        os.makedirs(fdir, exist_ok=True)
        with open(os.path.join(fdir, image_filename), "wb") as fh:
            fh.write(image_bytes)
        image_url = f"saved_patches/{folder_name}/{image_filename}".replace("\\", "/")

        health_ids = labels_to_ids(health_labels, HEALTH_LABEL_ID)
        stressor_ids = labels_to_ids(stressor_labels, STRESSOR_LABEL_ID)
        ai_ids = labels_to_ids(ai_health_labels or [], HEALTH_LABEL_ID)
        # annotations.csv `label` = every applicable label id: coral health
        # (1=Healthy, 2=Compromised, 3=Dead) plus any human-added stressors
        # (4-8), de-duplicated and numerically sorted, e.g. "2,6".
        _parts = [p for p in (health_ids.split(",") if health_ids else [])
                  + (stressor_ids.split(",") if stressor_ids else []) if p]
        label_combined = ",".join(sorted(set(_parts), key=lambda s: int(s)))
        if source is None:
            source = "ai_accepted" if set(ai_ids.split(",")) == set(health_ids.split(",")) else "human_corrected"
        depth_num = _num(meta.get("depth"))
        temp_num = _num(meta.get("temp"))
        surveyid = str(meta.get("surveyid") or survey_date or folder_name).strip()
        now = _dt.now().isoformat(timespec="seconds")

        # 2) upsert the survey
        con.execute(
            "INSERT OR IGNORE INTO surveys "
            "(surveyid, transectid, survey_date, site, site_fullname, folder_name, "
            " lat_start, lng_start, camera, depth, temp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (surveyid, str(meta.get("transectid", "")), survey_date, site,
             str(meta.get("site_fullname", "")), folder_name, str(meta.get("lat_start", "")),
             str(meta.get("lng_start", "")), str(meta.get("camera", "")),
             str(meta.get("depth", "")), str(meta.get("temp", ""))))

        # 3) insert the coral_dataset patch row
        con.execute(
            "INSERT INTO coral_dataset "
            "(image_filename, image_url, patchid, site_from_filename, survey_date_from_filename, "
            " surveyid, survey_date, site, folder_name, coral_health_labels, stressor_labels, "
            " depth, temp, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (image_filename, image_url, patchid, site, survey_date, surveyid, survey_date, site,
             folder_name, health_ids, stressor_ids, depth_num, temp_num, now))

        # 4) insert the annotation event
        annotation_id = _next_annotation_id(con)
        con.execute(
            "INSERT INTO annotations "
            "(annotation_id, image_filename, patchid, label, surveyid, survey_date, site, source, "
            " ai_health_labels, human_health_labels, coral_health_labels, stressor_labels, "
            " annotated_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (annotation_id, image_filename, patchid, label_combined, surveyid, survey_date, site, source,
             ai_ids, health_ids, health_ids, stressor_ids, annotated_by, now))
        con.commit()
    finally:
        con.close()

    # 5) append denormalised patch row to coral_dataset_cleaned.csv
    patch_row = {
        "image_filename": image_filename, "image_url": image_url, "patchid": patchid,
        "site_from_filename": site, "survey_date_from_filename": survey_date,
        "surveyid": surveyid, "transectid": str(meta.get("transectid", "")),
        "survey_date": survey_date, "site": site, "site_fullname": str(meta.get("site_fullname", "")),
        "folder_name": folder_name, "lat_start": str(meta.get("lat_start", "")),
        "lng_start": str(meta.get("lng_start", "")), "camera": str(meta.get("camera", "")),
        "depth": depth_num if depth_num is not None else "",
        "temp": temp_num if temp_num is not None else "",
        "coral_health_labels": health_ids, "stressor_labels": stressor_ids,
    }
    _append_csv_row(CSV_PATH, PATCH_COLUMNS, patch_row)

    # 6) append survey row to surveys_metadata.csv only if this surveyid is new
    existing_surveyids = _existing_surveyids()
    print(f"[DEBUG] Checking if surveyid '{surveyid}' exists in: {existing_surveyids}")

    if surveyid and surveyid not in existing_surveyids:
        # Read existing CSV to check structure
        file_exists = os.path.exists(METADATA_CSV_PATH)
    
        # Read the current content to check the header
        if file_exists:
            with open(METADATA_CSV_PATH, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"[DEBUG] Current file content:\n{content}")
    
        # Prepare row as a simple comma-separated string to ensure no formatting issues
        row_values = [
            str(surveyid),  # surveyid - make sure it's a string
            str(meta.get("transectid", "")),
            str(survey_date),
            str(site),
            str(meta.get("site_fullname", "")),
            str(folder_name),
            str(meta.get("lat_start", "")),
            str(meta.get("lng_start", "")),
            str(meta.get("camera", "")),
            str(meta.get("depth", "")),
            str(meta.get("temp", "")),
            ]
    
        print(f"[DEBUG] Row values to write: {row_values}")
        print(f"[DEBUG] Number of columns: {len(row_values)}, expected: {len(SURVEY_COLUMNS)}")
        # Write directly with manual quoting to avoid issues
        os.makedirs(os.path.dirname(METADATA_CSV_PATH), exist_ok=True)
        with open(METADATA_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            if not file_exists:
                # Write header
                f.write(",".join(SURVEY_COLUMNS) + "\n")
        
            # Write row - manually join with commas
            # Use csv.QUOTE_MINIMAL to handle any special characters
            import csv
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(row_values)
            f.flush()
    
        print(f"[DEBUG] Successfully appended surveyid '{surveyid}'")
    
    # Verify by reading back the last line
        with open(METADATA_CSV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 1:
                print(f"[DEBUG] Last line: {lines[-1].strip()}")
    else:
        print(f"[DEBUG] Surveyid '{surveyid}' already exists or is empty, skipping append")

    # 7) append annotation row to annotations.csv (two columns: patchid,label)
    #    patchid = uploaded image name; label = health(1-3) + stressor(4-8) ids
    _append_csv_row(ANNOTATIONS_CSV_PATH, ANNOTATIONS_CSV_COLUMNS,
                    {"patchid": patchid, "label": label_combined})

    clear_data_caches()
    return {
        "image_filename": image_filename, "patchid": patchid, "image_url": image_url,
        "surveyid": surveyid, "survey_date": survey_date, "folder_name": folder_name,
        "coral_health_labels": health_ids, "stressor_labels": stressor_ids,
        "annotation_id": annotation_id, "source": source,
        "db_path": DB_PATH, "csv_path": CSV_PATH, "metadata_path": METADATA_CSV_PATH,
        "annotations_path": ANNOTATIONS_CSV_PATH,
    }


def db_counts():
    """(surveys, coral_dataset patches, annotations) row counts from the live DB."""
    if not os.path.exists(DB_PATH):
        return 0, 0, 0
    con = sqlite3.connect(DB_PATH)
    try:
        return _count(con, "surveys"), _count(con, "coral_dataset"), _count(con, "annotations")
    finally:
        con.close()


def clear_data_caches():
    """Invalidate cached dataset/site aggregations so new saves show up live."""
    for fn in (load_dataset, build_site_geo):
        try:
            fn.clear()
        except Exception:
            pass