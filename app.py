from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException, BadRequest
import os
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Loaders
import pickle
import joblib
import gzip
import bz2
import lzma

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ROOT = Path(__file__).resolve().parent
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "1") == "1"  # set to 0 in prod


def smart_load_model(path: Path):
    """
    Try multiple ways to load a serialized model:
      1) joblib.load (works for joblib and many pickles, including compressed)
      2) raw pickle.load
      3) gzip/bz2/lzma + pickle.load
    """
    last_err = None
    try:
        return joblib.load(path)
    except Exception as e:
        last_err = e
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_err = e
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_err = e
    try:
        with bz2.open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_err = e
    try:
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        last_err = e
    raise last_err


def try_load(path, name):
    try:
        model = smart_load_model(path)
        app.logger.info(f"Loaded {name} from {path}")
        return model, None
    except Exception as e:
        app.logger.exception(f"Failed to load {name} from {path}: {e}")
        return None, str(e)


# Load models
rainfall_model, rainfall_err = try_load(ROOT / "rainfall_model.pkl", "rainfall_model")
cost_model, cost_err = try_load(ROOT / "cost_estimation_model.pkl", "cost_estimation_model")
feasibility_model, feasibility_err = try_load(ROOT / "feasibility_model.pkl", "feasibility_model")
structure_model, structure_err = try_load(ROOT / "structure_model.pkl", "structure_model")

# Feature names based on actual model training
RAINFALL_FEATURES = ["location_lat", "location_lng"]
COST_FEATURES = ["StructureTypeEncoded", "dwellers"]  # Model expects encoded structure type
FEASIBILITY_FEATURES = ["rainfall", "roof_area", "open_space"]
STRUCTURE_FEATURES = ["annual_rainfall", "roof_area", "open_space"]

# Structure type encoding mapping (based on your training data)
STRUCTURE_TYPE_ENCODING = {
    "Pit": 0,
    "pit": 0,  # case insensitive
    "PIT": 0,
    "Tank": 1,
    "tank": 1,
    "TANK": 1
    # Add other structure types if they exist in your training data
}


def get_expected_features_from_model(model):
    """
    Best-effort introspection of scikit-learn models/pipelines to see what features they expect.
    """
    names = []
    try:
        # scikit-learn 1.0+ estimators often expose feature_names_in_
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        # XGBoost models
        if hasattr(model, "feature_names") and model.feature_names:
            return list(model.feature_names)
        # Pipelines: try the final step
        if hasattr(model, "steps"):
            last = model.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
            if hasattr(last, "feature_names") and last.feature_names:
                return list(last.feature_names)
        # ColumnTransformer or preprocessor may hold get_feature_names_out
        if hasattr(model, "get_feature_names_out"):
            return list(model.get_feature_names_out())
    except Exception:
        pass
    return names or None


# Log introspected features at startup (helps debugging)
def log_model_expectations():
    for name, m in [
        ("rainfall_model", rainfall_model),
        ("cost_model", cost_model),
        ("feasibility_model", feasibility_model),
        ("structure_model", structure_model),
    ]:
        if m is None:
            app.logger.warning(f"{name}: not loaded ({locals().get(name + '_err', '')})")
            continue
        feats = get_expected_features_from_model(m)
        app.logger.info(f"{name} expected features: {feats}")


log_model_expectations()


def parse_payload(required_columns, input_mapping=None):
    """
    Parse request payload and optionally map input field names to model field names.

    Args:
        required_columns: List of column names the model expects
        input_mapping: Dict mapping user input names to model names (optional)
    """
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("Invalid or missing JSON body. Ensure Content-Type: application/json and valid JSON.")

    rows = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(x, dict) for x in rows):
        raise BadRequest("Payload must be an object or an array of objects.")

    df = pd.DataFrame(rows)

    # Apply input mapping if provided
    if input_mapping:
        df = df.rename(columns=input_mapping)
        # Check for mapped columns
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            original_names = [k for k, v in input_mapping.items() if v in missing]
            raise BadRequest(
                f"Missing required fields: {original_names} (maps to {missing}). Provided keys: {list(payload.keys() if isinstance(payload, dict) else payload[0].keys())}")
    else:
        # Check for exact column names
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise BadRequest(f"Missing required fields: {missing}. Provided keys: {list(df.columns)}")

    # Reorder to match model training
    df = df[required_columns]
    return df


def encode_structure_type(df):
    """
    Encode structure_type to StructureTypeEncoded based on training mapping.
    """
    if 'structure_type' in df.columns:
        # Apply encoding
        df['StructureTypeEncoded'] = df['structure_type'].map(STRUCTURE_TYPE_ENCODING)

        # Check for unknown structure types
        unknown_mask = df['StructureTypeEncoded'].isna()
        if unknown_mask.any():
            unknown_types = df.loc[unknown_mask, 'structure_type'].unique().tolist()
            valid_types = list(STRUCTURE_TYPE_ENCODING.keys())
            raise BadRequest(f"Unknown structure_type values: {unknown_types}. Valid values: {valid_types}")

        # Remove original column and keep only encoded
        df = df.drop('structure_type', axis=1)

    return df


def to_jsonable(preds):
    if isinstance(preds, (np.ndarray, pd.Series)):
        preds = preds.tolist()
    elif hasattr(preds, "item"):
        preds = preds.item()
    if isinstance(preds, list):
        return [p.item() if hasattr(p, "item") else p for p in preds]
    if hasattr(preds, "item"):
        return preds.item()
    return preds


def error_response(e, status=500):
    app.logger.exception(e)
    body = {
        "error": type(e).__name__,
        "message": str(e) if DEBUG_ERRORS else "An unexpected error occurred.",
        "status_code": status,
    }
    if DEBUG_ERRORS:
        body["traceback"] = traceback.format_exc()
    return jsonify(body), status


@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException):
    body = {
        "error": e.name,
        "message": e.description,
        "status_code": e.code,
    }
    return jsonify(body), e.code


@app.errorhandler(Exception)
def handle_unexpected_exception(e: Exception):
    return error_response(e, 500)


# ==============================================================================
# HOME AND STATUS ENDPOINTS
# ==============================================================================

@app.route("/")
def home():
    return """
    <h1>🌧️ Rainwater Harvesting System API</h1>
    <p>Welcome to the Rainwater Harvesting Prediction API!</p>

    <h2>📋 Available Endpoints:</h2>
    <h3>🌐 Browser-Friendly (GET):</h3>
    <ul>
        <li><a href="/help">📖 Help & Documentation</a></li>
        <li><a href="/models">🔧 Model Status</a></li>
        <li><a href="/predict_rainfall_get?lat=26.83&lng=80.86">🌧️ Rainfall Prediction Demo</a></li>
        <li><a href="/predict_cost_get?structure_type=Pit&dwellers=14">💰 Cost Estimation Demo</a></li>
        <li><a href="/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83">✅ Feasibility Check Demo</a></li>
        <li><a href="/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4">🏗️ Structure Type Demo</a></li>
    </ul>

    <h3>📡 API Endpoints (POST):</h3>
    <ul>
        <li><code>POST /predict_rainfall</code> - Rainfall prediction</li>
        <li><code>POST /predict_cost</code> - Cost estimation</li>
        <li><code>POST /predict_feasibility</code> - Feasibility analysis</li>
        <li><code>POST /predict_structure</code> - Structure type recommendation</li>
    </ul>

    <p><em>Built by karthi281104 - 2025</em></p>
    """


@app.route("/healthz")
def health():
    return jsonify({"status": "ok", "timestamp": "2025-09-20 18:35:36"}), 200


@app.route("/models")
def models_status():
    def status(model, err):
        return "loaded" if model else f"error: {err}"

    return jsonify({
        "rainfall_model": status(rainfall_model, rainfall_err),
        "rainfall_expected_features": get_expected_features_from_model(rainfall_model),
        "cost_model": status(cost_model, cost_err),
        "cost_expected_features": get_expected_features_from_model(cost_model),
        "feasibility_model": status(feasibility_model, feasibility_err),
        "feasibility_expected_features": get_expected_features_from_model(feasibility_model),
        "structure_model": status(structure_model, structure_err),
        "structure_expected_features": get_expected_features_from_model(structure_model),
        "structure_type_encoding": STRUCTURE_TYPE_ENCODING
    }), 200


@app.route("/help")
def help_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rainwater Harvesting API - Help</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .url { background: #e9ecef; padding: 8px; border-radius: 3px; font-family: monospace; }
            code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🌧️ Rainwater Harvesting System API Documentation</h1>

        <h2>🌐 Browser-Friendly GET Endpoints</h2>
        <p>Click these links to test directly in your browser:</p>

        <div class="endpoint">
            <h3>🌧️ Rainfall Prediction</h3>
            <span class="method">GET</span>
            <div class="url">/predict_rainfall_get?lat={latitude}&lng={longitude}</div>
            <p><strong>Example:</strong> <a href="/predict_rainfall_get?lat=26.83&lng=80.86">/predict_rainfall_get?lat=26.83&lng=80.86</a></p>
            <p><strong>Parameters:</strong> lat (float), lng (float)</p>
        </div>

        <div class="endpoint">
            <h3>💰 Cost Estimation</h3>
            <span class="method">GET</span>
            <div class="url">/predict_cost_get?structure_type={type}&dwellers={number}</div>
            <p><strong>Example:</strong> <a href="/predict_cost_get?structure_type=Pit&dwellers=14">/predict_cost_get?structure_type=Pit&dwellers=14</a></p>
            <p><strong>Parameters:</strong> structure_type (Pit/Tank), dwellers (integer)</p>
        </div>

        <div class="endpoint">
            <h3>✅ Feasibility Analysis</h3>
            <span class="method">GET</span>
            <div class="url">/predict_feasibility_get?rainfall={amount}&roof_area={area}&open_space={space}</div>
            <p><strong>Example:</strong> <a href="/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83">/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83</a></p>
            <p><strong>Parameters:</strong> rainfall (float), roof_area (float), open_space (float)</p>
        </div>

        <div class="endpoint">
            <h3>🏗️ Structure Type Recommendation</h3>
            <span class="method">GET</span>
            <div class="url">/predict_structure_get?annual_rainfall={amount}&roof_area={area}&open_space={space}</div>
            <p><strong>Example:</strong> <a href="/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4">/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4</a></p>
            <p><strong>Parameters:</strong> annual_rainfall (float), roof_area (float), open_space (float)</p>
        </div>

        <h2>📡 POST API Endpoints</h2>
        <p>For use with Postman, curl, or frontend applications:</p>

        <div class="endpoint">
            <h3>🌧️ POST /predict_rainfall</h3>
            <p><strong>Body (JSON):</strong></p>
            <pre><code>{"lat": 26.83, "lng": 80.86}</code></pre>
        </div>

        <div class="endpoint">
            <h3>💰 POST /predict_cost</h3>
            <p><strong>Body (JSON):</strong></p>
            <pre><code>{"structure_type": "Pit", "dwellers": 14}</code></pre>
        </div>

        <div class="endpoint">
            <h3>✅ POST /predict_feasibility</h3>
            <p><strong>Body (JSON):</strong></p>
            <pre><code>{"rainfall": 1326, "roof_area": 71, "open_space": 83}</code></pre>
        </div>

        <div class="endpoint">
            <h3>🏗️ POST /predict_structure</h3>
            <p><strong>Body (JSON):</strong></p>
            <pre><code>{"annual_rainfall": 336, "roof_area": 15, "open_space": 4}</code></pre>
        </div>

        <h2>🔧 Utility Endpoints</h2>
        <ul>
            <li><a href="/models">📊 Model Status & Features</a> - Check if models are loaded</li>
            <li><a href="/healthz">❤️ Health Check</a> - API status</li>
        </ul>

        <p><em>API developed by karthi281104 - September 2025</em></p>
    </body>
    </html>
    """


# ==============================================================================
# POST ENDPOINTS (Original API)
# ==============================================================================

@app.route("/predict_rainfall", methods=["POST"])
def predict_rainfall():
    """
    Predicts rainfall based on latitude and longitude.
    Accepts: {"lat": 26.83, "lng": 80.86}
    Maps to: {"location_lat": 26.83, "location_lng": 80.86}
    """
    try:
        if not rainfall_model:
            return jsonify({"error": "Model not available", "details": rainfall_err}), 503

        # Map user-friendly input names to model names
        input_mapping = {"lat": "location_lat", "lng": "location_lng"}
        df = parse_payload(RAINFALL_FEATURES, input_mapping)
        preds = to_jsonable(rainfall_model.predict(df))
        if len(df) == 1 and isinstance(preds, list):
            preds = preds[0]
        return jsonify({"rainfall_prediction": preds})
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_cost", methods=["POST"])
def predict_cost():
    """
    Predicts cost estimation based on structure type and number of dwellers.
    Accepts: {"structure_type": "Pit", "dwellers": 14}
    Automatically encodes structure_type to StructureTypeEncoded
    """
    try:
        if not cost_model:
            return jsonify({"error": "Model not available", "details": cost_err}), 503

        # Parse input (expecting structure_type and dwellers)
        payload = request.get_json(silent=True)
        if payload is None:
            raise BadRequest("Invalid or missing JSON body. Ensure Content-Type: application/json and valid JSON.")

        rows = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(x, dict) for x in rows):
            raise BadRequest("Payload must be an object or an array of objects.")

        df = pd.DataFrame(rows)

        # Check required input fields
        required_input = ["structure_type", "dwellers"]
        missing = [c for c in required_input if c not in df.columns]
        if missing:
            raise BadRequest(f"Missing required fields: {missing}. Provided keys: {list(df.columns)}")

        # Encode structure_type to StructureTypeEncoded
        df = encode_structure_type(df)

        # Reorder to match model expectation
        df = df[COST_FEATURES]

        preds = to_jsonable(cost_model.predict(df))
        if len(df) == 1 and isinstance(preds, list):
            preds = preds[0]
        return jsonify({"cost_estimation_prediction": preds})
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_feasibility", methods=["POST"])
def predict_feasibility():
    """
    Predicts feasibility based on rainfall, roof area, and open space.
    Accepts: {"rainfall": 1326, "roof_area": 71, "open_space": 83}
    Returns: Feasibility prediction (likely "Yes"/"No" or probability)
    """
    try:
        if not feasibility_model:
            return jsonify({"error": "Model not available", "details": feasibility_err}), 503
        df = parse_payload(FEASIBILITY_FEATURES)
        preds = to_jsonable(feasibility_model.predict(df))
        if len(df) == 1 and isinstance(preds, list):
            preds = preds[0]
        return jsonify({"feasibility_prediction": preds})
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_structure", methods=["POST"])
def predict_structure():
    """
    Predicts structure type based on annual rainfall, roof area, and open space.
    Accepts: {"annual_rainfall": 336, "roof_area": 15, "open_space": 4}
    Expected to return structure type (based on your data, likely returns codes that map to structure types)
    """
    try:
        if not structure_model:
            return jsonify({"error": "Model not available", "details": structure_err}), 503
        df = parse_payload(STRUCTURE_FEATURES)
        preds = to_jsonable(structure_model.predict(df))
        if len(df) == 1 and isinstance(preds, list):
            preds = preds[0]
        return jsonify({"structure_type_prediction": preds})
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(e, 400)


# ==============================================================================
# GET ENDPOINTS (Browser-Friendly)
# ==============================================================================

@app.route("/predict_rainfall_get", methods=["GET"])
def predict_rainfall_get():
    """
    GET version for easy browser testing
    Usage: /predict_rainfall_get?lat=26.83&lng=80.86
    """
    try:
        if not rainfall_model:
            return jsonify({"error": "Model not available", "details": rainfall_err}), 503

        # Get parameters from URL
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)

        if lat is None or lng is None:
            return jsonify({
                "error": "Missing parameters",
                "message": "Required: lat and lng",
                "example": "/predict_rainfall_get?lat=26.83&lng=80.86"
            }), 400

        # Create DataFrame with correct column names
        df = pd.DataFrame({
            'location_lat': [lat],
            'location_lng': [lng]
        })

        preds = to_jsonable(rainfall_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        return jsonify({
            "rainfall_prediction": preds,
            "input": {"lat": lat, "lng": lng}
        })
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_cost_get", methods=["GET"])
def predict_cost_get():
    """
    GET version for easy browser testing
    Usage: /predict_cost_get?structure_type=Pit&dwellers=14
    """
    try:
        if not cost_model:
            return jsonify({"error": "Model not available", "details": cost_err}), 503

        structure_type = request.args.get('structure_type', type=str)
        dwellers = request.args.get('dwellers', type=int)

        if structure_type is None or dwellers is None:
            return jsonify({
                "error": "Missing parameters",
                "message": "Required: structure_type and dwellers",
                "example": "/predict_cost_get?structure_type=Pit&dwellers=14"
            }), 400

        # Encode structure type
        if structure_type not in STRUCTURE_TYPE_ENCODING:
            valid_types = list(STRUCTURE_TYPE_ENCODING.keys())
            return jsonify({
                "error": "Invalid structure_type",
                "message": f"Valid values: {valid_types}",
                "provided": structure_type
            }), 400

        encoded_type = STRUCTURE_TYPE_ENCODING[structure_type]

        # Create DataFrame
        df = pd.DataFrame({
            'StructureTypeEncoded': [encoded_type],
            'dwellers': [dwellers]
        })

        preds = to_jsonable(cost_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        return jsonify({
            "cost_estimation_prediction": preds,
            "input": {"structure_type": structure_type, "dwellers": dwellers, "encoded_type": encoded_type}
        })
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_feasibility_get", methods=["GET"])
def predict_feasibility_get():
    """
    GET version for easy browser testing
    Usage: /predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83
    """
    try:
        if not feasibility_model:
            return jsonify({"error": "Model not available", "details": feasibility_err}), 503

        rainfall = request.args.get('rainfall', type=float)
        roof_area = request.args.get('roof_area', type=float)
        open_space = request.args.get('open_space', type=float)

        if rainfall is None or roof_area is None or open_space is None:
            return jsonify({
                "error": "Missing parameters",
                "message": "Required: rainfall, roof_area, open_space",
                "example": "/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83"
            }), 400

        df = pd.DataFrame({
            'rainfall': [rainfall],
            'roof_area': [roof_area],
            'open_space': [open_space]
        })

        preds = to_jsonable(feasibility_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        return jsonify({
            "feasibility_prediction": preds,
            "input": {"rainfall": rainfall, "roof_area": roof_area, "open_space": open_space}
        })
    except Exception as e:
        return error_response(e, 400)


@app.route("/predict_structure_get", methods=["GET"])
def predict_structure_get():
    """
    GET version for easy browser testing
    Usage: /predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4
    """
    try:
        if not structure_model:
            return jsonify({"error": "Model not available", "details": structure_err}), 503

        annual_rainfall = request.args.get('annual_rainfall', type=float)
        roof_area = request.args.get('roof_area', type=float)
        open_space = request.args.get('open_space', type=float)

        if annual_rainfall is None or roof_area is None or open_space is None:
            return jsonify({
                "error": "Missing parameters",
                "message": "Required: annual_rainfall, roof_area, open_space",
                "example": "/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4"
            }), 400

        df = pd.DataFrame({
            'annual_rainfall': [annual_rainfall],
            'roof_area': [roof_area],
            'open_space': [open_space]
        })

        preds = to_jsonable(structure_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        return jsonify({
            "structure_type_prediction": preds,
            "input": {"annual_rainfall": annual_rainfall, "roof_area": roof_area, "open_space": open_space}
        })
    except Exception as e:
        return error_response(e, 400)


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "1") == "1"

    print("🌧️ Starting Rainwater Harvesting System API...")
    print(f"📍 Server: http://{host}:{port}")
    print(f"📖 Help: http://{host}:{port}/help")
    print(f"🔧 Models: http://{host}:{port}/models")
    print("=" * 50)

    app.run(host=host, port=port, debug=debug)
