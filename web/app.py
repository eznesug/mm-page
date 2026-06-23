from pathlib import Path
import sys

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_figures_final import (
    OXIDANT_E0,
    ensure_web_inference_assets,
    predict_with_conformal_bundle,
)

MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)


def load_runtime_assets():
    assets = ensure_web_inference_assets(MODEL_DIR)
    feature_groups = {
        "Core descriptors": [feat for feat in assets["features"] if feat != "Oxidant_E0"],
    }
    return {
        "model": assets["model"],
        "scaler": assets["scaler"],
        "features": assets["features"],
        "conformal_bundle": assets["conformal_bundle"],
        "metadata": assets["metadata"],
        "feature_groups": feature_groups,
    }


RUNTIME = load_runtime_assets()


def build_feature_vector(form_data, features):
    oxidant = form_data.get("oxidant", "").strip()
    if oxidant not in OXIDANT_E0:
        raise ValueError("Invalid oxidant selection.")

    feature_values = []
    field_values = {}
    for feat in features:
        if feat == "Oxidant_E0":
            feature_values.append(float(OXIDANT_E0[oxidant]))
            continue

        raw_value = form_data.get(feat, "").strip()
        if raw_value == "":
            raise ValueError(f"Missing feature: {feat}")

        value = float(raw_value)
        feature_values.append(value)
        field_values[feat] = raw_value

    return oxidant, np.asarray(feature_values, dtype=float).reshape(1, -1), field_values


def run_prediction(feature_vector):
    x_scaled = RUNTIME["scaler"].transform(feature_vector)
    point_prediction = float(RUNTIME["model"].predict(x_scaled)[0])
    conformal_result = predict_with_conformal_bundle(RUNTIME["conformal_bundle"], x_scaled)

    intervals = {}
    for level, values in sorted(conformal_result["intervals"].items()):
        intervals[level] = {
            "lower": float(values["lower"][0]),
            "upper": float(values["upper"][0]),
            "width": float(values["width"][0]),
        }

    return {
        "point_prediction": point_prediction,
        "model_name": RUNTIME["metadata"].get("model_name", "Best model"),
        "sigma": float(conformal_result["sigma"][0]),
        "intervals": intervals,
    }


@app.route("/")
def welcome():
    return render_template("welcome.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    form_values = {"oxidant": "OH", "smiles": ""}

    if request.method == "POST":
        try:
            oxidant, feature_vector, field_values = build_feature_vector(request.form, RUNTIME["features"])
            form_values.update(field_values)
            form_values["oxidant"] = oxidant
            form_values["smiles"] = request.form.get("smiles", "").strip()
            prediction = run_prediction(feature_vector)
        except Exception as exc:
            error = str(exc)
            for feat in RUNTIME["features"]:
                if feat != "Oxidant_E0":
                    form_values[feat] = request.form.get(feat, "").strip()
            form_values["oxidant"] = request.form.get("oxidant", "OH").strip() or "OH"
            form_values["smiles"] = request.form.get("smiles", "").strip()

    return render_template(
        "predict.html",
        feature_groups=RUNTIME["feature_groups"],
        intervals=sorted(prediction["intervals"].keys()) if prediction else [],
        prediction=prediction,
        error=error,
        form_values=form_values,
    )


if __name__ == "__main__":
    app.run(debug=True)
