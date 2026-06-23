import os
import warnings
from pathlib import Path
from math import pi

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


EXCEL_CANDIDATES = [
    Path(os.getenv("QSAR_EXCEL_FILE", "")),
    # Path(r"C:\Users\win10\Desktop\data_with_alkane.xlsx"),
    # Path(r"D:\Users\hkbg\Desktop\data_with_alkane.xlsx"),
    # Path(r"D:\Users\hkbg\Desktop\新建文件夹 (2)\data_with_alkane.xlsx"),
    # Path(r"/data_with_alkane.xlsx"),
    Path(r"data_with_alkane.xlsx"),
    Path(r"data_with_alkane.xlsx"),
    Path(r"data_with_alkane.xlsx"),
    Path(r"data_with_alkane.xlsx"),
]

RESULT_DIR = Path("paper_aligned_results")
SHAP_DIR = RESULT_DIR / "SHAP"
AD_DIR = RESULT_DIR / "Applicability_Domain"
MODEL_DIR = RESULT_DIR / "model_artifacts"

SYSTEMS = {"OH": "logkOH", "O3": "logkO3", "NO3": "logkNO3"}
OXIDANT_E0 = {"OH": 2.80, "O3": 2.07, "NO3": 2.40}

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

C_PRIMARY = "#1B4F72"
C_ACCENT = "#2E86C1"
C_LIGHT = "#AED6F1"
C_ORANGE = "#E67E22"
C_ORANGE_L = "#FAD7A0"
C_GREEN = "#1E8449"
C_RED = "#C0392B"
C_GRAY = "#566573"

MODEL_COLORS = {"Ridge": "#2E86C1", "SVR": "#E67E22", "RF": "#1E8449", "GBDT": "#8E44AD"}
MODEL_MARKERS = {"Ridge": "o", "SVR": "s", "RF": "^", "GBDT": "D"}
SYS_COLORS = {"OH": C_PRIMARY, "O3": C_ORANGE, "NO3": C_GREEN}

CMAP_BLUE = LinearSegmentedColormap.from_list("blue_pub", ["#FDFEFE", "#AED6F1", "#1B4F72"])
CMAP_BWOR = LinearSegmentedColormap.from_list("bwor_pub", ["#1B4F72", "#FDFEFE", "#E67E22"])

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#BDC3C7",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def ensure_dirs():
    for folder in [RESULT_DIR, SHAP_DIR, AD_DIR, MODEL_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def resolve_excel_file() -> Path:
    for path in EXCEL_CANDIDATES:
        if not path:
            continue
        path_str = str(path).strip()
        if not path_str or path_str == ".":
            continue
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError("Could not find data_with_alkane.xlsx. Please set QSAR_EXCEL_FILE.")


def evaluate_model(y_true, y_pred, name=""):
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }
    print(f"{name} | R2={metrics['R2']:.3f}, RMSE={metrics['RMSE']:.3f}, MAE={metrics['MAE']:.3f}")
    return metrics


def add_panel_label(ax, label, x=-0.14, y=1.06, fontsize=13):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", va="top", ha="left")


def add_polar_label_by_gs(fig, gs, row, col, label, fontsize=13):
    sp = gs.get_subplot_params()
    nrows, ncols = gs.get_geometry()
    total_w = sp.right - sp.left
    total_h = sp.top - sp.bottom
    col_w = total_w / (ncols + sp.wspace * (ncols - 1))
    row_h = total_h / (nrows + sp.hspace * (nrows - 1))
    col_gap = col_w * sp.wspace
    row_gap = row_h * sp.hspace
    cell_left = sp.left + col * (col_w + col_gap)
    cell_top = sp.top - row * (row_h + row_gap)
    fig.text(cell_left - 0.14 * col_w, cell_top + 0.06 * row_h, label, fontsize=fontsize, fontweight="bold",
             va="top", ha="left", transform=fig.transFigure)


def normalize_high(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo + 1e-9) for v in vals]


def normalize_low(vals):
    lo, hi = min(vals), max(vals)
    return [1 - (v - lo) / (hi - lo + 1e-9) for v in vals]


def safe_group_metrics(y_true, y_pred, group_flag):
    out = {}
    for group_name, mask in {"Alkane": group_flag, "Non-alkane": ~group_flag}.items():
        if np.sum(mask) < 2:
            out[group_name] = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": int(np.sum(mask))}
        else:
            out[group_name] = {
                "R2": r2_score(y_true[mask], y_pred[mask]),
                "RMSE": np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])),
                "MAE": mean_absolute_error(y_true[mask], y_pred[mask]),
                "n": int(np.sum(mask)),
            }
    return out


def print_metric_block(title, metrics_dict):
    print(f"\n{title}")
    for model_name, metrics in metrics_dict.items():
        print(
            f"  {model_name:<6} | "
            f"R2={metrics['R2']:.3f}, "
            f"RMSE={metrics['RMSE']:.3f}, "
            f"MAE={metrics['MAE']:.3f}"
        )


def print_group_metric_block(title, group_metrics_dict):
    print(f"\n{title}")
    for model_name, groups in group_metrics_dict.items():
        print(f"  {model_name}:")
        for group_name in ["Alkane", "Non-alkane"]:
            metrics = groups[group_name]
            if np.isnan(metrics["R2"]):
                print(f"    {group_name:<12} | n={metrics['n']}, insufficient samples")
            else:
                print(
                    f"    {group_name:<12} | "
                    f"n={metrics['n']}, "
                    f"R2={metrics['R2']:.3f}, "
                    f"RMSE={metrics['RMSE']:.3f}, "
                    f"MAE={metrics['MAE']:.3f}"
                )


def williams_ad_arrays(X_train, y_train, X_test, y_test, model):
    xtr = np.asarray(X_train, dtype=float)
    xte = np.asarray(X_test, dtype=float)
    xtr_aug = np.column_stack([np.ones(xtr.shape[0]), xtr])
    xte_aug = np.column_stack([np.ones(xte.shape[0]), xte])
    xtx_inv = np.linalg.pinv(xtr_aug.T @ xtr_aug)
    lev_tr = np.einsum("ij,jk,ik->i", xtr_aug, xtx_inv, xtr_aug)
    lev_te = np.einsum("ij,jk,ik->i", xte_aug, xtx_inv, xte_aug)
    resid_train = y_train - model.predict(X_train)
    sigma = np.sqrt(np.sum(resid_train ** 2) / max(len(y_train) - xtr_aug.shape[1], 1))
    sigma = max(float(sigma), 1e-12)
    sres_tr = resid_train / np.sqrt(np.maximum(sigma ** 2 * (1 - lev_tr), 1e-12))
    resid_test = y_test - model.predict(X_test)
    sres_te = resid_test / np.sqrt(np.maximum(sigma ** 2 * (1 - lev_te), 1e-12))
    h_star = 3 * xtr_aug.shape[1] / xtr_aug.shape[0]
    return lev_tr, sres_tr, lev_te, sres_te, h_star


def make_shap_explainer(model, x_background):
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        return shap.TreeExplainer(model)
    background = shap.sample(x_background, min(100, len(x_background)), random_state=RANDOM_STATE)
    return shap.Explainer(model.predict, background)


def get_shap_values(explainer, x_data):
    out = explainer(x_data)
    return np.asarray(out.values) if hasattr(out, "values") else np.asarray(out)


def load_combined_dataset(excel_file: Path):
    dfs = []
    system_frames = {}
    for sys_name, target in SYSTEMS.items():
        df = pd.read_excel(excel_file, sheet_name=sys_name)
        descriptor_cols = [
            c for c in df.columns
            if c not in ["SMILES", "molecule_names", "is_alkane"] and not c.startswith("logk")
        ]
        df = df.dropna(subset=descriptor_cols + [target]).copy()
        df["System"] = sys_name
        df["Oxidant_E0"] = OXIDANT_E0[sys_name]
        df["Target"] = df[target]
        df["is_alkane"] = df["is_alkane"].astype(str).str.lower().isin(["1", "true", "yes", "y", "alkane"])
        system_frames[sys_name] = df.copy()
        dfs.append(df[["molecule_names", "is_alkane", "System"] + descriptor_cols + ["Oxidant_E0", "Target"]])
    data = pd.concat(dfs, axis=0).reset_index(drop=True)
    feature_cols = [c for c in data.columns if c not in ["Target", "molecule_names", "is_alkane", "System"]]
    return data, feature_cols, system_frames


def split_and_scale(data, feature_cols):
    x = data[feature_cols].values
    y = data["Target"].values
    names = data["molecule_names"].values
    alk = data["is_alkane"].values.astype(bool)
    systems = data["System"].values

    x_train, x_tmp, y_train, y_tmp, names_train, names_tmp, alk_train, alk_tmp, sys_train, sys_tmp = train_test_split(
        x, y, names, alk, systems, train_size=TRAIN_SIZE, random_state=RANDOM_STATE, stratify=systems
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    x_val, x_test, y_val, y_test, names_val, names_test, alk_val, alk_test, sys_val, sys_test = train_test_split(
        x_tmp, y_tmp, names_tmp, alk_tmp, sys_tmp, train_size=val_ratio, random_state=RANDOM_STATE, stratify=sys_tmp
    )

    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    return {
        "X_train": x_train_s,
        "X_val": x_val_s,
        "X_test": x_test_s,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "names_train": names_train,
        "names_val": names_val,
        "names_test": names_test,
        "alk_train": alk_train,
        "alk_val": alk_val,
        "alk_test": alk_test,
        "sys_train": sys_train,
        "sys_val": sys_val,
        "sys_test": sys_test,
        "scaler": scaler,
    }


def train_models(split_data):
    models = {
        "Ridge": Ridge(alpha=1.0),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1),
        "RF": RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE),
        "GBDT": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE),
    }

    metrics_train = {}
    metrics_val = {}
    metrics_test = {}
    preds_train = {}
    preds_val = {}
    preds_test = {}
    test_group_metrics = {}
    for model_name, model in models.items():
        model.fit(split_data["X_train"], split_data["y_train"])
        y_train_pred = model.predict(split_data["X_train"])
        y_val_pred = model.predict(split_data["X_val"])
        y_test_pred = model.predict(split_data["X_test"])
        metrics_train[model_name] = evaluate_model(split_data["y_train"], y_train_pred, f"Train-{model_name}")
        metrics_val[model_name] = evaluate_model(split_data["y_val"], y_val_pred, f"Val-{model_name}")
        metrics_test[model_name] = evaluate_model(split_data["y_test"], y_test_pred, f"Test-{model_name}")
        preds_train[model_name] = y_train_pred
        preds_val[model_name] = y_val_pred
        preds_test[model_name] = y_test_pred
        test_group_metrics[model_name] = safe_group_metrics(split_data["y_test"], y_test_pred, split_data["alk_test"])

    print_metric_block("Training Set Metrics", metrics_train)
    print_metric_block("Validation Set Metrics", metrics_val)
    print_metric_block("Test Set Metrics", metrics_test)
    print_group_metric_block("Test Set Group Metrics", test_group_metrics)

    best_name = max(metrics_val, key=lambda k: metrics_val[k]["R2"])
    best_model = models[best_name]
    preds_best = {
        "train": preds_train[best_name],
        "val": preds_val[best_name],
        "test": preds_test[best_name],
    }
    print(f"\nBest model: {best_name}")
    evaluate_model(split_data["y_train"], preds_best["train"], f"{best_name}-Train")
    evaluate_model(split_data["y_val"], preds_best["val"], f"{best_name}-Val")
    evaluate_model(split_data["y_test"], preds_best["test"], f"{best_name}-Test")
    return models, best_name, best_model, metrics_val, preds_val, test_group_metrics, preds_best


def plot_figure1_dataset_overview(data, split_data, system_frames, df_imp):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.44, wspace=0.40, left=0.07, right=0.97, top=0.93, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    for sys_name in SYSTEMS:
        vals = system_frames[sys_name][SYSTEMS[sys_name]].values
        ax.hist(vals, bins=22, density=True, alpha=0.20, color=SYS_COLORS[sys_name], edgecolor="none")
        x_kde = np.linspace(vals.min() - 0.5, vals.max() + 0.5, 300)
        ax.plot(x_kde, gaussian_kde(vals)(x_kde), color=SYS_COLORS[sys_name], lw=2.2, label=f"{sys_name} ($n$={len(vals)})")
    ax.set_xlabel("log$k$ (M$^{-1}$s$^{-1}$)")
    ax.set_ylabel("Density")
    ax.set_title("log$k$ Distribution by Oxidant System")
    ax.legend(fontsize=8.2)
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    box_data_alk, box_data_nalk = [], []
    for sys_name, target in SYSTEMS.items():
        df = system_frames[sys_name]
        box_data_alk.append(df.loc[df["is_alkane"], target].values)
        box_data_nalk.append(df.loc[~df["is_alkane"], target].values)
    ax.boxplot(box_data_alk, positions=[1, 4, 7], widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=C_ORANGE_L, linewidth=1.1), medianprops=dict(color=C_RED, linewidth=2.0),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.boxplot(box_data_nalk, positions=[2, 5, 8], widths=0.7, patch_artist=True,
               boxprops=dict(facecolor=C_LIGHT, linewidth=1.1), medianprops=dict(color=C_RED, linewidth=2.0),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.set_xticks([1.5, 4.5, 7.5])
    ax.set_xticklabels(list(SYSTEMS.keys()))
    ax.set_ylabel("log$k$")
    ax.set_title("log$k$ by System and Compound Type")
    ax.legend(handles=[mpatches.Patch(facecolor=C_ORANGE_L, edgecolor="gray", label="Alkane"),
                       mpatches.Patch(facecolor=C_LIGHT, edgecolor="gray", label="Non-alkane")], fontsize=8)
    add_panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[0, 2])
    n_total = len(data)
    n_alk = int(data["is_alkane"].sum())
    n_nalk = n_total - n_alk
    n_train, n_val, n_test = len(split_data["y_train"]), len(split_data["y_val"]), len(split_data["y_test"])
    ax.pie([n_train, n_val, n_test], colors=[C_PRIMARY, C_ORANGE, C_GREEN], radius=1.0, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2))
    ax.pie([n_alk, n_nalk], colors=[C_ORANGE_L, C_LIGHT], radius=0.65, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.30, edgecolor="white", linewidth=2))
    ax.text(0, 0, f"$n$={n_total}", ha="center", va="center", fontsize=11, fontweight="bold", color=C_PRIMARY)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_PRIMARY, label=f"Train {n_train}"),
        mpatches.Patch(facecolor=C_ORANGE, label=f"Val {n_val}"),
        mpatches.Patch(facecolor=C_GREEN, label=f"Test {n_test}"),
        mpatches.Patch(facecolor=C_ORANGE_L, label=f"Alkane {n_alk}"),
        mpatches.Patch(facecolor=C_LIGHT, label=f"Non-alkane {n_nalk}"),
    ], fontsize=7.4, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Dataset Composition")
    add_panel_label(ax, "(c)", x=-0.08)

    ax = fig.add_subplot(gs[1, 0])
    e0_vals = [OXIDANT_E0[s] for s in SYSTEMS]
    mean_vals = [system_frames[s][SYSTEMS[s]].mean() for s in SYSTEMS]
    std_vals = [system_frames[s][SYSTEMS[s]].std() for s in SYSTEMS]
    for i, sys_name in enumerate(SYSTEMS):
        ax.errorbar(e0_vals[i], mean_vals[i], yerr=std_vals[i], fmt="o", color=SYS_COLORS[sys_name], ms=10, lw=1.5, capsize=5, capthick=1.5)
        ax.text(e0_vals[i] + 0.02, mean_vals[i] + std_vals[i] + 0.05, sys_name, fontsize=9, color=SYS_COLORS[sys_name], fontweight="bold")
    slope, intercept, r_val, _, _ = stats.linregress(e0_vals, mean_vals)
    x_line = np.linspace(min(e0_vals) - 0.1, max(e0_vals) + 0.1, 100)
    ax.plot(x_line, slope * x_line + intercept, color=C_GRAY, lw=1.2, ls="--", alpha=0.7)
    ax.text(0.05, 0.08, f"$r$ = {r_val:.2f}", transform=ax.transAxes, fontsize=8.5, color=C_GRAY)
    ax.set_xlabel("Oxidant reduction potential $E^0$ (V)")
    ax.set_ylabel("Mean log$k$ ± SD")
    ax.set_title("Oxidant Reactivity vs. Reduction Potential")
    add_panel_label(ax, "(d)")

    ax = fig.add_subplot(gs[1, 1])
    resid_train = np.abs(split_data["y_train"] - split_data["y_train_pred"])
    resid_test = np.abs(split_data["y_test"] - split_data["y_test_pred"])
    for resid, color, label in [(resid_train, C_PRIMARY, "Train"), (resid_test, C_ORANGE, "Test")]:
        x_sorted = np.sort(resid)
        y_cum = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        ax.plot(x_sorted, y_cum, color=color, lw=2.0, label=label)
        p90 = np.percentile(x_sorted, 90)
        ax.axvline(p90, color=color, lw=0.9, ls=":", alpha=0.7)
        ax.text(p90 + 0.02, 0.15, f"P90={p90:.2f}", color=color, fontsize=7.5, rotation=90, va="bottom")
    ax.axhline(0.90, color=C_GRAY, lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("|Residual|")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Cumulative Absolute Error Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    add_panel_label(ax, "(e)")

    ax = fig.add_subplot(gs[1, 2])
    top_corr = min(10, len(df_imp))
    top_features = df_imp.head(top_corr)["Feature"].tolist()
    idx = [split_data["features"].index(f) for f in top_features]
    corr_matrix = np.corrcoef(split_data["X_train"][:, idx].T)
    corr_display = np.where(np.triu(np.ones_like(corr_matrix, dtype=bool), k=1), np.nan, corr_matrix)
    im = ax.imshow(corr_display, cmap=CMAP_BWOR, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(top_corr))
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(top_corr))
    ax.set_yticklabels(top_features, fontsize=7)
    ax.tick_params(length=0)
    for i in range(top_corr):
        for j in range(top_corr):
            if not np.isnan(corr_display[i, j]):
                tcolor = "white" if abs(corr_display[i, j]) > 0.6 else C_PRIMARY
                ax.text(j, i, f"{corr_display[i, j]:.2f}", ha="center", va="center", fontsize=6.2, color=tcolor)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Pearson $r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Feature Correlation (Top Descriptors)")
    add_panel_label(ax, "(f)")

    fig.savefig(RESULT_DIR / "Fig1_Dataset_Overview.png")
    plt.close(fig)


def plot_figure2_model_performance(metrics_all, best_model_name, split_data, preds_val, preds_best):
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38, left=0.07, right=0.97, top=0.93, bottom=0.08)
    model_names = list(metrics_all.keys())
    r2s = [metrics_all[m]["R2"] for m in model_names]
    rmses = [metrics_all[m]["RMSE"] for m in model_names]
    maes = [metrics_all[m]["MAE"] for m in model_names]

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(model_names))
    width = 0.24
    bars1 = ax.bar(x - width, r2s, width, label="$R^2$", color=C_PRIMARY, alpha=0.88)
    bars2 = ax.bar(x, rmses, width, label="RMSE", color=C_ORANGE, alpha=0.88)
    bars3 = ax.bar(x + width, maes, width, label="MAE", color=C_GREEN, alpha=0.88)
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Validation Set")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_ylim(0, max(r2s) * 1.22)
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    metric_matrix = np.array([[metrics_all[m][k] for m in model_names] for k in ["R2", "RMSE", "MAE"]])
    norm_matrix = np.zeros_like(metric_matrix)
    for i, row in enumerate(metric_matrix):
        lo, hi = row.min(), row.max()
        if hi - lo < 1e-9:
            norm_matrix[i] = 0.5
        elif i == 0:
            norm_matrix[i] = (row - lo) / (hi - lo)
        else:
            norm_matrix[i] = 1 - (row - lo) / (hi - lo)
    im = ax.imshow(norm_matrix, cmap=CMAP_BLUE, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["$R^2$", "RMSE", "MAE"])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(len(model_names)):
            color = "white" if norm_matrix[i, j] > 0.55 else C_PRIMARY
            ax.text(j, i, f"{metric_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Normalized score", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Performance Heat Map")
    add_panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[0, 2], polar=True)
    angles = np.linspace(0, 2 * pi, 3, endpoint=False).tolist() + [0]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["$R^2$ (↑)", "RMSE (↓)", "MAE (↓)"], fontsize=8.5)
    r2_n, rmse_n, mae_n = normalize_high(r2s), normalize_low(rmses), normalize_low(maes)
    for i, model_name in enumerate(model_names):
        vals = [r2_n[i], rmse_n[i], mae_n[i], r2_n[i]]
        ax.plot(angles, vals, color=MODEL_COLORS[model_name], lw=1.8, label=model_name)
        ax.fill(angles, vals, color=MODEL_COLORS[model_name], alpha=0.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.18), fontsize=7.5, framealpha=0.9)
    ax.set_title("Normalized Radar Chart", pad=16)

    ax = fig.add_subplot(gs[1, 0])
    lo = min(split_data["y_val"].min(), min(v.min() for v in preds_val.values())) - 0.3
    hi = max(split_data["y_val"].max(), max(v.max() for v in preds_val.values())) + 0.3
    for model_name, y_pred in preds_val.items():
        ax.scatter(split_data["y_val"], y_pred, color=MODEL_COLORS[model_name], marker=MODEL_MARKERS[model_name],
                   s=22, alpha=0.65, label=model_name, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=C_RED, lw=1.2, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed log$k$")
    ax.set_ylabel("Predicted log$k$")
    ax.set_title("Validation Parity for All Models")
    ax.legend(fontsize=7.5)
    ax.set_aspect("equal", "box")
    add_panel_label(ax, "(d)")

    ax = fig.add_subplot(gs[1, 1])
    all_obs = np.concatenate([split_data["y_train"], split_data["y_val"], split_data["y_test"]])
    all_pred = np.concatenate([preds_best["train"], preds_best["val"], preds_best["test"]])
    lo = min(all_obs.min(), all_pred.min()) - 0.3
    hi = max(all_obs.max(), all_pred.max()) + 0.3
    for label, yt, yp, col, mk, sz, alp in [
        ("Train", split_data["y_train"], preds_best["train"], C_PRIMARY, "o", 18, 0.5),
        ("Val", split_data["y_val"], preds_best["val"], C_ORANGE, "s", 22, 0.65),
        ("Test", split_data["y_test"], preds_best["test"], C_GREEN, "^", 25, 0.8),
    ]:
        ax.scatter(yt, yp, color=col, marker=mk, s=sz, alpha=alp, edgecolors="none",
                   label=f"{label} ($R^2$={r2_score(yt, yp):.3f})")
    ax.plot([lo, hi], [lo, hi], color=C_RED, lw=1.2, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed log$k$")
    ax.set_ylabel("Predicted log$k$")
    ax.set_title(f"Parity Plot for Best Model ({best_model_name})")
    ax.legend(fontsize=7)
    ax.set_aspect("equal", "box")
    add_panel_label(ax, "(e)")

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(preds_best["train"], split_data["y_train"] - preds_best["train"], color=C_PRIMARY, s=15, alpha=0.45, edgecolors="none", label="Train")
    ax.scatter(preds_best["test"], split_data["y_test"] - preds_best["test"], color=C_ORANGE, s=22, alpha=0.75, edgecolors="none", label="Test", marker="^")
    ax.axhline(0, color=C_RED, lw=1.2, ls="--")
    ax.axhline(2, color="#BDC3C7", lw=0.8, ls=":")
    ax.axhline(-2, color="#BDC3C7", lw=0.8, ls=":")
    ax.set_xlabel("Predicted log$k$")
    ax.set_ylabel("Residual (Obs - Pred)")
    ax.set_title("Residual Scatter Plot")
    ax.legend(fontsize=7.5)
    add_panel_label(ax, "(f)")

    add_polar_label_by_gs(fig, gs, row=0, col=2, label="(c)")
    fig.savefig(RESULT_DIR / "Fig2_Model_Comparison.png")
    plt.close(fig)


def plot_figure3_external_validation(best_model_name, split_data, preds_best):
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.34, left=0.07, right=0.97, top=0.90, bottom=0.12)

    y_test = split_data["y_test"]
    y_test_pred = preds_best["test"]
    alk_mask = split_data["alk_test"]
    nalk_mask = ~alk_mask
    residuals = y_test - y_test_pred

    ax = fig.add_subplot(gs[0, 0])
    lo = min(y_test.min(), y_test_pred.min()) - 0.3
    hi = max(y_test.max(), y_test_pred.max()) + 0.3
    ax.scatter(y_test, y_test_pred, color=C_PRIMARY, s=28, alpha=0.72, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=C_RED, lw=1.2, ls="--")
    ax.text(
        0.05, 0.93,
        f"$R^2$ = {r2_score(y_test, y_test_pred):.3f}\nRMSE = {np.sqrt(mean_squared_error(y_test, y_test_pred)):.3f}\nMAE = {mean_absolute_error(y_test, y_test_pred):.3f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.92)
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed log$k$")
    ax.set_ylabel("Predicted log$k$")
    ax.set_title(f"Overall Test Parity ({best_model_name})")
    ax.set_aspect("equal", "box")
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(y_test[nalk_mask], y_test_pred[nalk_mask], color=C_PRIMARY, s=28, alpha=0.68, edgecolors="none",
               label=f"Non-alkane ($n$={nalk_mask.sum()})")
    ax.scatter(y_test[alk_mask], y_test_pred[alk_mask], color=C_ORANGE, marker="^", s=40, alpha=0.85, edgecolors="none",
               label=f"Alkane ($n$={alk_mask.sum()})")
    ax.plot([lo, hi], [lo, hi], color=C_RED, lw=1.2, ls="--")
    r2_alk = r2_score(y_test[alk_mask], y_test_pred[alk_mask]) if alk_mask.sum() >= 2 else np.nan
    r2_nalk = r2_score(y_test[nalk_mask], y_test_pred[nalk_mask]) if nalk_mask.sum() >= 2 else np.nan
    ax.text(
        0.05, 0.93,
        f"Non-alkane $R^2$={r2_nalk:.3f}\nAlkane $R^2$={r2_alk:.3f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.92)
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed log$k$")
    ax.set_ylabel("Predicted log$k$")
    ax.set_title("Test Parity by Compound Type")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_aspect("equal", "box")
    add_panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[0, 2])
    groups = ["Non-alkane", "Alkane"]
    parts = ax.violinplot([residuals[nalk_mask], residuals[alk_mask]], positions=range(len(groups)),
                          showmedians=False, showextrema=False, widths=0.62)
    pal = {"Alkane": C_ACCENT, "Non-alkane": C_ORANGE}
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(pal[g]); pc.set_alpha(0.35); pc.set_edgecolor(pal[g])
    ax.boxplot([residuals[nalk_mask], residuals[alk_mask]], positions=range(len(groups)), widths=0.18, patch_artist=True,
               boxprops=dict(facecolor="white", linewidth=1.2), medianprops=dict(color=C_RED, linewidth=2),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.axhline(0, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Prediction Error")
    ax.set_title("External Validation Error Distribution")
    add_panel_label(ax, "(c)")

    fig.savefig(RESULT_DIR / "Fig3_External_Validation.png")
    plt.close(fig)


def plot_figure4_applicability_domain(best_model, best_model_name, split_data, preds_best):
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.34, left=0.07, right=0.97, top=0.90, bottom=0.12)
    resid_train = split_data["y_train"] - preds_best["train"]
    resid_test = split_data["y_test"] - preds_best["test"]

    ax = fig.add_subplot(gs[0, 0])
    lev_tr, sres_tr, lev_te, sres_te, h_star = williams_ad_arrays(
        split_data["X_train"], split_data["y_train"], split_data["X_test"], split_data["y_test"], best_model
    )
    mask_tr_in = (lev_tr <= h_star) & (np.abs(sres_tr) <= 3)
    mask_te_in = (lev_te <= h_star) & (np.abs(sres_te) <= 3)
    ax.scatter(lev_tr[mask_tr_in], sres_tr[mask_tr_in], color=C_PRIMARY, s=20, alpha=0.55, edgecolors="none", label="Train (in AD)")
    ax.scatter(lev_tr[~mask_tr_in], sres_tr[~mask_tr_in], color=C_PRIMARY, s=35, alpha=0.90, edgecolors=C_PRIMARY, marker="x", linewidths=1.2, label="Train (out)")
    ax.scatter(lev_te[mask_te_in], sres_te[mask_te_in], color=C_ORANGE, s=28, alpha=0.70, edgecolors="none", marker="s", label="Test (in AD)")
    ax.scatter(lev_te[~mask_te_in], sres_te[~mask_te_in], color=C_RED, s=45, alpha=0.95, edgecolors=C_RED, marker="s", linewidths=1.2, label="Test (out)")
    ax.axvline(h_star, color=C_RED, lw=1.4, ls="--", label=f"$h^*$ = {h_star:.3f}")
    ax.axhline(3, color=C_GRAY, lw=0.9, ls=":")
    ax.axhline(-3, color=C_GRAY, lw=0.9, ls=":")
    ax.set_xlabel("Leverage $h$")
    ax.set_ylabel("Standardized Residual")
    ax.set_title(f"Williams Plot of {best_model_name}")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(-5.5, 5.5)
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(lev_tr.max(), lev_te.max()) * 1.05, 24)
    ax.hist(lev_tr, bins=bins, color=C_PRIMARY, alpha=0.35, density=True, label="Train")
    ax.hist(lev_te, bins=bins, color=C_ORANGE, alpha=0.35, density=True, label="Test")
    ax.axvline(h_star, color=C_RED, lw=1.3, ls="--", label=f"$h^*$={h_star:.3f}")
    ax.set_xlabel("Leverage $h$")
    ax.set_ylabel("Density")
    ax.set_title("Leverage Distribution")
    ax.legend(fontsize=7.5)
    add_panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(min(sres_tr.min(), sres_te.min()) - 0.2, max(sres_tr.max(), sres_te.max()) + 0.2, 28)
    ax.hist(sres_tr, bins=bins, color=C_PRIMARY, alpha=0.35, density=True, label="Train")
    ax.hist(sres_te, bins=bins, color=C_ORANGE, alpha=0.35, density=True, label="Test")
    ax.axvline(3, color=C_RED, lw=1.2, ls=":")
    ax.axvline(-3, color=C_RED, lw=1.2, ls=":")
    ax.axvline(0, color=C_GRAY, lw=1.0, ls="--", alpha=0.8)
    ax.set_xlabel("Standardized Residual")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution in AD Space")
    ax.legend(fontsize=7.5)
    add_panel_label(ax, "(c)")

    fig.savefig(RESULT_DIR / "Fig4_Applicability_Domain.png")
    plt.close(fig)

    pd.DataFrame({"Molecule": split_data["names_train"], "Leverage": lev_tr, "StdResid": sres_tr,
                  "Out_of_AD": (lev_tr > h_star) | (np.abs(sres_tr) > 3)}).to_csv(AD_DIR / "Train_Out_of_AD.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"Molecule": split_data["names_test"], "Leverage": lev_te, "StdResid": sres_te,
                  "Out_of_AD": (lev_te > h_star) | (np.abs(sres_te) > 3)}).to_csv(AD_DIR / "Test_Out_of_AD.csv", index=False, encoding="utf-8-sig")


def plot_figure4_shap(best_model, split_data, df_imp, df_compare):
    explainer = make_shap_explainer(best_model, split_data["X_train"])
    shap_values_train = get_shap_values(explainer, split_data["X_train"])
    shap_values_test = get_shap_values(explainer, split_data["X_test"])

    fig = plt.figure(figsize=(13.5, 8.4))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.32, left=0.08, right=0.97, top=0.93, bottom=0.08)

    top_n = min(12, len(split_data["features"]))
    ax = fig.add_subplot(gs[0, 0])
    top_df = df_imp.head(top_n)
    colors = CMAP_BLUE(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax.barh(range(top_n), top_df["Importance"].values, color=colors, edgecolor="none", height=0.65)
    for bar in bars:
        ax.text(bar.get_width() + top_df["Importance"].max() * 0.015, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", fontsize=7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["Feature"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_n} Feature Importance")
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    top_shap = min(10, len(split_data["features"]))
    top_idx = df_imp.head(top_shap).index.tolist()
    feat_order = list(reversed(top_idx))
    rng = np.random.default_rng(RANDOM_STATE)
    for yi, fi in enumerate(feat_order):
        sv = shap_values_train[:, fi]
        fval = split_data["X_train"][:, fi]
        fv_norm = (fval - fval.min()) / (fval.max() - fval.min() + 1e-9)
        jitter = rng.uniform(-0.25, 0.25, size=len(sv))
        sc = ax.scatter(sv, yi + jitter, c=fv_norm, cmap=CMAP_BWOR, s=8, alpha=0.55, vmin=0, vmax=1, edgecolors="none")
    ax.axvline(0, color=C_GRAY, lw=0.8, ls="--")
    ax.set_yticks(range(top_shap))
    ax.set_yticklabels([split_data["features"][i] for i in feat_order], fontsize=7.5)
    ax.set_xlabel("SHAP value")
    ax.set_title("SHAP Beeswarm Plot")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Feature value\n(low → high)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    top_comp = min(10, len(df_compare))
    comp_df = df_compare.head(top_comp)
    cx = np.arange(top_comp)
    cw = 0.35
    ax.barh(cx - cw / 2, comp_df["Alkane"].values, cw, color=C_ORANGE, alpha=0.85, label="Alkane", edgecolor="none")
    ax.barh(cx + cw / 2, comp_df["Non_alkane"].values, cw, color=C_PRIMARY, alpha=0.85, label="Non-alkane", edgecolor="none")
    ax.set_yticks(cx)
    ax.set_yticklabels(comp_df["Feature"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP: Alkane vs Non-alkane")
    ax.legend(fontsize=8, loc="lower right")
    add_panel_label(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    top_heat = min(10, len(split_data["features"]))
    top_samp = min(50, len(shap_values_test))
    top_names = df_imp.head(top_heat)["Feature"].tolist()
    top_idx_h = [split_data["features"].index(f) for f in top_names]
    shap_heat = shap_values_test[:top_samp, :][:, top_idx_h]
    shap_heat_norm = shap_heat / (np.abs(shap_heat).max(axis=0, keepdims=True) + 1e-9)
    im = ax.imshow(shap_heat_norm.T, aspect="auto", cmap=CMAP_BWOR, vmin=-1, vmax=1)
    ax.set_yticks(range(top_heat))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel(f"Test sample index (first {top_samp})")
    ax.set_title("SHAP Value Heatmap")
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for ai in np.where(split_data["alk_test"][:top_samp])[0]:
        ax.axvline(ai, color=C_ORANGE, lw=0.7, alpha=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Norm. SHAP", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    add_panel_label(ax, "(d)", x=-0.04)

    fig.savefig(RESULT_DIR / "Fig5_SHAP_Analysis.png")
    plt.close(fig)

    df_imp.to_csv(SHAP_DIR / "SHAP_feature_importance.csv", index=False, encoding="utf-8-sig")
    df_compare.to_csv(SHAP_DIR / "SHAP_Alkane_vs_Nonalkane.csv", index=False, encoding="utf-8-sig")


def plot_figure6_alkane_validation(group_metrics_all, split_data, preds_best):
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.34, left=0.07, right=0.97, top=0.90, bottom=0.12)

    y_test = split_data["y_test"]
    y_test_pred = preds_best["test"]
    alk_mask = split_data["alk_test"]
    nalk_mask = ~alk_mask
    residuals = y_test - y_test_pred

    ax = fig.add_subplot(gs[0, 0])
    groups = ["Non-alkane", "Alkane"]
    parts = ax.violinplot([residuals[nalk_mask], residuals[alk_mask]], positions=range(len(groups)),
                          showmedians=False, showextrema=False, widths=0.62)
    pal = {"Alkane": C_ACCENT, "Non-alkane": C_ORANGE}
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(pal[g]); pc.set_alpha(0.35); pc.set_edgecolor(pal[g])
    ax.boxplot([residuals[nalk_mask], residuals[alk_mask]], positions=range(len(groups)), widths=0.18, patch_artist=True,
               boxprops=dict(facecolor="white", linewidth=1.2), medianprops=dict(color=C_RED, linewidth=2),
               whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    ax.axhline(0, color=C_RED, lw=1.0, ls="--", alpha=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Prediction Error")
    ax.set_title("Error Distribution by Compound Type")
    add_panel_label(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1], polar=True)
    angles = np.linspace(0, 2 * pi, 3, endpoint=False).tolist() + [0]
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["$R^2$", "1-RMSE*", "1-MAE*"], fontsize=8)
    all_r2 = [g[k]["R2"] for g in group_metrics_all.values() for k in g if not np.isnan(g[k]["R2"])]
    all_rmse = [g[k]["RMSE"] for g in group_metrics_all.values() for k in g if not np.isnan(g[k]["RMSE"])]
    all_mae = [g[k]["MAE"] for g in group_metrics_all.values() for k in g if not np.isnan(g[k]["MAE"])]
    r2_min, r2_max = min(all_r2), max(all_r2)
    rmse_min, rmse_max = min(all_rmse), max(all_rmse)
    mae_min, mae_max = min(all_mae), max(all_mae)
    for i, (model_name, gm) in enumerate(group_metrics_all.items()):
        col = list(MODEL_COLORS.values())[i]
        for group_name, line_style in [("Non-alkane", "-"), ("Alkane", "--")]:
            if np.isnan(gm[group_name]["R2"]):
                continue
            vals = [
                (gm[group_name]["R2"] - r2_min) / max(r2_max - r2_min, 1e-9),
                1 - (gm[group_name]["RMSE"] - rmse_min) / max(rmse_max - rmse_min, 1e-9),
                1 - (gm[group_name]["MAE"] - mae_min) / max(mae_max - mae_min, 1e-9),
            ]
            vals += vals[:1]
            ax.plot(angles, vals, color=col, ls=line_style, lw=2.0, label=f"{model_name} ({group_name})", alpha=0.85)
            ax.fill(angles, vals, color=col, alpha=0.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.10, 1.12), fontsize=5.5, ncol=2, framealpha=0.92)
    ax.set_title("Alkane vs Non-alkane\n(All Models)", pad=16, fontsize=10)

    ax = fig.add_subplot(gs[0, 2])
    lo = min(y_test[alk_mask].min(), y_test_pred[alk_mask].min()) - 0.3
    hi = max(y_test[alk_mask].max(), y_test_pred[alk_mask].max()) + 0.3
    ax.scatter(y_test[alk_mask], y_test_pred[alk_mask], color=C_ORANGE, marker="^", s=46, alpha=0.88, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=C_RED, lw=1.2, ls="--")
    if alk_mask.sum() >= 2:
        ax.text(
            0.05, 0.93,
            f"Alkane $R^2$={r2_score(y_test[alk_mask], y_test_pred[alk_mask]):.3f}\nRMSE={np.sqrt(mean_squared_error(y_test[alk_mask], y_test_pred[alk_mask])):.3f}\nMAE={mean_absolute_error(y_test[alk_mask], y_test_pred[alk_mask]):.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BDC3C7", alpha=0.92)
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed log$k$")
    ax.set_ylabel("Predicted log$k$")
    ax.set_title("Alkane-only Test Parity")
    ax.set_aspect("equal", "box")
    add_panel_label(ax, "(c)")

    add_polar_label_by_gs(fig, gs, row=0, col=1, label="(b)")
    fig.savefig(RESULT_DIR / "Fig6_Alkane_Validation.png")
    plt.close(fig)


def main():
    ensure_dirs()
    excel_file = resolve_excel_file()
    print(f"Using Excel file: {excel_file}")

    data, features, system_frames = load_combined_dataset(excel_file)
    split_data = split_and_scale(data, features)
    split_data["features"] = features

    _, best_name, best_model, metrics_val, preds_val, group_metrics_all, preds_best = train_models(split_data)
    split_data["y_train_pred"] = preds_best["train"]
    split_data["y_test_pred"] = preds_best["test"]

    explainer = make_shap_explainer(best_model, split_data["X_train"])
    shap_values_train = get_shap_values(explainer, split_data["X_train"])
    shap_values_test = get_shap_values(explainer, split_data["X_test"])
    shap_importance = np.abs(shap_values_train).mean(axis=0)
    df_imp = pd.DataFrame({"Feature": features, "Importance": shap_importance}).sort_values("Importance", ascending=False)
    df_compare = pd.DataFrame({
        "Feature": features,
        "Alkane": np.abs(shap_values_test[split_data["alk_test"]]).mean(axis=0),
        "Non_alkane": np.abs(shap_values_test[~split_data["alk_test"]]).mean(axis=0),
    })
    df_compare["Diff"] = df_compare["Alkane"] - df_compare["Non_alkane"]
    df_compare = df_compare.sort_values("Diff", ascending=False)

    plot_figure1_dataset_overview(data, split_data, system_frames, df_imp)
    plot_figure2_model_performance(metrics_val, best_name, split_data, preds_val, preds_best)
    plot_figure3_external_validation(best_name, split_data, preds_best)
    plot_figure4_applicability_domain(best_model, best_name, split_data, preds_best)
    plot_figure4_shap(best_model, split_data, df_imp, df_compare)
    plot_figure6_alkane_validation(group_metrics_all, split_data, preds_best)

    joblib.dump(best_model, MODEL_DIR / "unified_qsar_model.pkl")
    joblib.dump(split_data["scaler"], MODEL_DIR / "scaler.pkl")
    with open(MODEL_DIR / "feature_order.txt", "w", encoding="utf-8") as f:
        for feat in features:
            f.write(feat + "\n")

    def plot_figure7_alkane_comparison(split_data, preds_best):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))

        # ===== 数据准备 =====
        y_test = split_data["y_test"]
        y_pred = preds_best["test"]
        alk_mask = split_data["alk_test"]
        names = split_data["names_test"]

        # ===== 本研究（Bias = Pred - Obs）=====
        df_ours = pd.DataFrame({
            "Molecule": names[alk_mask],
            "y_true": y_test[alk_mask],
            "y_pred": y_pred[alk_mask],
        })
        df_ours["bias"] = df_ours["y_pred"] - df_ours["y_true"]

        # ===== MLR =====
        df_mlr = pd.read_excel("MLR.xlsx")
        df_mlr["bias"] = df_mlr["y_pred"] - df_mlr["y_true"]

        # ===== 对齐 =====
        df_merge = pd.merge(df_ours, df_mlr, on="Molecule", suffixes=("_ours", "_mlr"))

        if len(df_merge) > 5:
            bias_ours = df_merge["bias_ours"].values
            bias_mlr = df_merge["bias_mlr"].values
        else:
            bias_ours = df_ours["bias"].values
            bias_mlr = df_mlr["bias"].values

        # ===== 统计（核心变化）=====
        mean_bias_ours = bias_ours.mean()
        mean_bias_mlr = bias_mlr.mean()

        std_ours = bias_ours.std()
        std_mlr = bias_mlr.std()

        improvement = (abs(mean_bias_mlr) - abs(mean_bias_ours)) / (abs(mean_bias_mlr) + 1e-9) * 100
        t_stat, p_val = stats.ttest_ind(bias_ours, bias_mlr, equal_var=False)

        # ===== 输出 =====
        print("\n========== Fig.7 (Mean Bias Comparison) ==========")
        print(f"Ours Mean Bias = {mean_bias_ours:.4f}")
        print(f"MLR Mean Bias  = {mean_bias_mlr:.4f}")
        print(f"|Bias| Improvement = {improvement:.2f}%")
        print(f"p-value = {p_val:.4e}")
        print("==================================================\n")

        # ===== violin =====
        parts = ax.violinplot([bias_ours, bias_mlr],
                              positions=[0, 1],
                              showextrema=False,
                              widths=0.6)

        for pc, col in zip(parts["bodies"], [C_PRIMARY, C_ORANGE]):
            pc.set_facecolor(col)
            pc.set_alpha(0.25)
            pc.set_edgecolor(col)

        # ===== box =====
        ax.boxplot([bias_ours, bias_mlr],
                   positions=[0, 1],
                   widths=0.18,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", linewidth=1.2),
                   medianprops=dict(color=C_RED, linewidth=2),
                   whiskerprops=dict(linewidth=1),
                   capprops=dict(linewidth=1))

        # ===== mean ± std =====
        ax.errorbar([0, 1],
                    [mean_bias_ours, mean_bias_mlr],
                    yerr=[std_ours, std_mlr],
                    fmt='o',
                    color='black',
                    capsize=4,
                    markersize=5,
                    zorder=3)

        # ===== 0基线（非常重要）=====
        ax.axhline(0, color=C_RED, lw=1.2, ls="--")

        # ===== 标注 =====
        y_max = max(bias_ours.max(), bias_mlr.max())
        y_min = min(bias_ours.min(), bias_mlr.min())

        ax.text(0.5, y_max * 1.05,
                f"Mean Bias Diff: {mean_bias_ours - mean_bias_mlr:.3f}",
                ha="center",
                fontsize=11,
                color=C_RED,
                fontweight="bold")

        # ===== 轴 =====
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["This work", "MLR"])
        ax.set_ylabel("Mean Bias (Pred - Obs)")
        ax.set_title("Prediction Bias Comparison for Alkanes")

        # ===== 美化 =====
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.savefig(RESULT_DIR / "Fig7_Mean_Bias.png")
        plt.close(fig)
    print("\nCompleted. Output files:")
    print(RESULT_DIR / "Fig1_Dataset_Overview.png")
    print(RESULT_DIR / "Fig2_Model_Comparison.png")
    print(RESULT_DIR / "Fig3_External_Validation.png")
    print(RESULT_DIR / "Fig4_Applicability_Domain.png")
    print(RESULT_DIR / "Fig5_SHAP_Analysis.png")
    print(RESULT_DIR / "Fig6_Alkane_Validation.png")
    plot_figure7_alkane_comparison(split_data, preds_best)


if __name__ == "__main__":
    main()
