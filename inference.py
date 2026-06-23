"""
Web 推理专用 — 从 paper_figures_final.py 剥离出的函数，
不含 matplotlib 依赖，仅用于 Flask 部署。
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

# ── 常量 ─────────────────────────────────────────────
OXIDANT_E0 = {"OH": 2.80, "O3": 2.07, "NO3": 2.40}


# ── 模型名称推断 ─────────────────────────────────────
def _infer_model_name(model):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR

    if isinstance(model, Ridge):
        return "Ridge"
    if isinstance(model, SVR):
        return "SVR"
    if isinstance(model, RandomForestRegressor):
        return "RF"
    if isinstance(model, GradientBoostingRegressor):
        return "GBDT"
    raise ValueError(f"Unsupported model type: {type(model)!r}")


# ── 加载已有推理资产 ─────────────────────────────────
def load_web_inference_assets(model_dir=None):
    source_dir = Path(model_dir)
    model = joblib.load(source_dir / "unified_qsar_model.pkl")
    scaler = joblib.load(source_dir / "scaler.pkl")
    conformal_bundle = joblib.load(source_dir / "conformal_predictor.pkl")

    with open(source_dir / "feature_order.txt", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    metadata_path = source_dir / "model_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "model_name": _infer_model_name(model),
            "feature_count": len(features),
            "interval_levels": sorted(
                int(level) for level in conformal_bundle["intervals"].keys()
            ),
        }

    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "conformal_bundle": conformal_bundle,
        "metadata": metadata,
        "model_dir": source_dir,
    }


# ── 确保资产就绪（只读路径） ─────────────────────────
def ensure_web_inference_assets(model_dir=None):
    source_dir = Path(model_dir)
    required_files = [
        source_dir / "unified_qsar_model.pkl",
        source_dir / "scaler.pkl",
        source_dir / "feature_order.txt",
        source_dir / "conformal_predictor.pkl",
        source_dir / "model_metadata.json",
    ]
    if all(path.exists() for path in required_files):
        return load_web_inference_assets(source_dir)

    # 如果模型文件不全，尝试从父目录 paper_figures_final 构建
    # （本地开发场景；生产环境应提前生成好模型文件）
    sys.path.insert(0, str(source_dir.parent))
    from paper_figures_final import build_and_save_web_inference_assets  # type: ignore

    build_and_save_web_inference_assets(model_dir=source_dir)
    return load_web_inference_assets(source_dir)


# ── 共形预测推理 ─────────────────────────────────────
def predict_with_conformal_bundle(conformal_bundle, X_scaled):
    x_scaled = np.asarray(X_scaled, dtype=float)
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(1, -1)

    fold_models = conformal_bundle["fold_models"]
    fold_preds = np.column_stack([model.predict(x_scaled) for model in fold_models])
    pred_mean = fold_preds.mean(axis=1)
    sigma = fold_preds.std(axis=1, ddof=0)
    beta = float(conformal_bundle.get("beta", 1.0))

    intervals = {}
    for level, interval_cfg in sorted(conformal_bundle["intervals"].items()):
        qhat = float(interval_cfg["qhat"])
        radius = qhat * (1.0 + beta * sigma)
        intervals[int(level)] = {
            "lower": pred_mean - radius,
            "upper": pred_mean + radius,
            "width": 2.0 * radius,
            "qhat": qhat,
        }

    return {
        "pred_mean": pred_mean,
        "sigma": sigma,
        "intervals": intervals,
    }
