from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import shap
import base64
import io
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables for models
model = None
explainer = None
feature_names = None


def load_models():
    """Load trained models"""
    global model, explainer, feature_names

    try:
        model_path = "Heart_Disease_Prediction/best_heart_disease_model.pkl"
        feature_path = "Heart_Disease_Prediction/feature_names.pkl"
        explainer_path = "Heart_Disease_Prediction/shap_explainer.pkl"

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✅ Model loaded successfully")
            # Test model with sample data
            test_input = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0]]
            test_pred = model.predict(test_input)
            print(f"✅ Model test prediction: {test_pred}")
        else:
            print(f"❌ Model file not found: {model_path}")

        if os.path.exists(feature_path):
            feature_names = joblib.load(feature_path)
            print(f"✅ Feature names loaded: {len(feature_names)} features")
            print(f"Features: {feature_names}")
        else:
            print(f"❌ Feature file not found: {feature_path}")

        if os.path.exists(explainer_path):
            explainer = joblib.load(explainer_path)
            print("✅ SHAP explainer loaded")
        else:
            print(f"❌ SHAP file not found: {explainer_path}")

    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback

        traceback.print_exc()


def create_shap_plot(shap_values, input_data, feature_names):
    """Create SHAP waterfall plot"""
    try:
        print(f"🎨 Creating SHAP plot with {len(shap_values)} values")
        # Clear any existing plots
        plt.clf()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sort by absolute SHAP values
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]

        # Prepare data
        y_pos = np.arange(len(sorted_idx))
        colors = ["#ff4444" if v < 0 else "#44ff44" for v in shap_values[sorted_idx]]

        # Create horizontal bar plot
        bars = ax.barh(y_pos, shap_values[sorted_idx], color=colors, alpha=0.7)

        # Customize
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{feature_names[i]}: {input_data[i]:.2f}" for i in sorted_idx]
        )
        ax.set_xlabel("SHAP Value (Impact on Prediction)")
        ax.set_title(
            "Feature Impact on Heart Disease Prediction", fontsize=14, fontweight="bold"
        )
        ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, shap_values[sorted_idx])):
            width = bar.get_width()
            ax.text(
                width + 0.01 if width >= 0 else width - 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                ha="left" if width >= 0 else "right",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()

        # Save to base64 string
        img = io.BytesIO()
        plt.savefig(img, format="png", dpi=150, bbox_inches="tight")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        print("✅ SHAP plot created successfully")
        return plot_url

    except Exception as e:
        print(f"❌ Error creating SHAP plot: {e}")
        import traceback

        traceback.print_exc()
        return None


@app.route("/")
def index():
    """Main page"""
    print("📄 Serving index page")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests"""
    print("\n🔍 === PREDICTION REQUEST STARTED ===")

    try:
        if model is None:
            print("❌ Model not loaded")
            return jsonify({"error": "Model not loaded"}), 500

        # Get input data from request
        data = request.json
        print(f"📊 Raw input data: {data}")

        # Extract features in correct order (11 features for your model)
        input_data = [
            float(data.get("age", 0)),
            float(data.get("sex", 0)),
            float(data.get("cp", 0)),
            float(data.get("trestbps", 0)),
            float(data.get("chol", 0)),
            float(data.get("fbs", 0)),
            float(data.get("restecg", 0)),
            float(data.get("thalach", 0)),
            float(data.get("exang", 0)),
            float(data.get("oldpeak", 0)),
            float(data.get("slope", 0)),
        ]

        print(f"📊 Processed input ({len(input_data)} features): {input_data}")

        # Make prediction
        print("🤖 Making prediction...")
        prediction = model.predict([input_data])[0]
        probability = model.predict_proba([input_data])[0]

        print(f"✅ Prediction: {prediction}")
        print(f"✅ Probability: {probability}")

        # Generate SHAP explanation if available
        shap_plot = None
        if explainer is not None and feature_names is not None:
            try:
                print("🔍 Generating SHAP explanation...")
                scaler = model.named_steps["scaler"]
                sample_scaled = scaler.transform([input_data])
                print(f"📊 Scaled input shape: {sample_scaled.shape}")

                shap_values = explainer.shap_values(sample_scaled)
                print(f"📊 SHAP values type: {type(shap_values)}")
                print(
                    f"📊 SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'No shape'}"
                )

                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    print(f"📊 SHAP is list with {len(shap_values)} elements")
                    if len(shap_values) > 1:
                        shap_values = shap_values[1]  # Use positive class
                    else:
                        shap_values = shap_values[0]

                if hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
                    shap_values = shap_values[0]  # Get first sample

                print(f"📊 Final SHAP values: {shap_values}")
                shap_plot = create_shap_plot(shap_values, input_data, feature_names)

            except Exception as e:
                print(f"⚠️ SHAP explanation error: {e}")
                import traceback

                traceback.print_exc()

        # Create response
        result = {
            "prediction": int(prediction),
            "probability": {
                "no_disease": float(probability[0]),
                "disease": float(probability[1]),
            },
            "confidence": float(max(probability)) * 100,
            "shap_plot": shap_plot,
        }

        print("✅ Response created, sending back to client")
        print(f"📊 Result keys: {result.keys()}")
        print("🔍 === PREDICTION REQUEST COMPLETED ===\n")

        return jsonify(result)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


@app.route("/health")
def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "feature_names_loaded": feature_names is not None,
        "feature_count": len(feature_names) if feature_names else 0,
    }
    print(f"🏥 Health check: {health_status}")
    return jsonify(health_status)


if __name__ == "__main__":
    print("🫀 Starting Enhanced Heart Disease Prediction Flask App...")
    load_models()

    print("\n📋 Startup Summary:")
    print(f"  Model loaded: {model is not None}")
    print(f"  Explainer loaded: {explainer is not None}")
    print(f"  Feature names loaded: {feature_names is not None}")
    print(f"  Expected features: 11")
    print("🚀 Flask app starting...\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
