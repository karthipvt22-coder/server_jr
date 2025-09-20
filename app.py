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

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "1") == "1"

def smart_load_model(path: Path):
    """
    Try multiple ways to load a serialized model.
    """
    errors = []
    
    # 1) joblib.load (most common for scikit-learn)
    try:
        model = joblib.load(path)
        print(f"✅ Loaded {path.name} with joblib")
        return model
    except Exception as e:
        errors.append(f"joblib: {e}")
    
    # 2) Standard pickle
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"✅ Loaded {path.name} with pickle")
        return model
    except Exception as e:
        errors.append(f"pickle: {e}")
    
    # 3) Skip compressed attempts for now since they're causing issues
    # If a file is actually compressed, we'll add specific handlers
    
    # Log all errors for debugging
    print(f"❌ Failed to load {path.name}:")
    for i, error in enumerate(errors, 1):
        print(f"  {i}. {error}")
    
    # Raise the first (most likely) error
    raise Exception(errors[0] if errors else "Unknown loading error")

def try_load(path, name):
    try:
        if not path.exists():
            return None, f"File not found: {path}"
        
        model = smart_load_model(path)
        app.logger.info(f"✅ Successfully loaded {name}")
        return model, None
    except Exception as e:
        error_msg = str(e)
        app.logger.error(f"❌ Failed to load {name}: {error_msg}")
        return None, error_msg

# Load models with better error handling
print("🔄 Loading ML models...")
rainfall_model, rainfall_err = try_load(ROOT / "rainfall_model.pkl", "rainfall_model")
cost_model, cost_err = try_load(ROOT / "cost_estimation_model.pkl", "cost_estimation_model") 
feasibility_model, feasibility_err = try_load(ROOT / "feasibility_model.pkl", "feasibility_model")
structure_model, structure_err = try_load(ROOT / "structure_model.pkl", "structure_model")

# Feature names based on your training data
RAINFALL_FEATURES = ["location_lat", "location_lng"]  
COST_FEATURES = ["StructureTypeEncoded", "dwellers"]
FEASIBILITY_FEATURES = ["rainfall", "roof_area", "open_space"]  
STRUCTURE_FEATURES = ["annual_rainfall", "roof_area", "open_space"]  

# Structure type encoding
STRUCTURE_TYPE_ENCODING = {
    "Pit": 0, "pit": 0, "PIT": 0,
    "Tank": 1, "tank": 1, "TANK": 1,
    "Pond": 2, "pond": 2, "POND": 2
}

def get_expected_features_from_model(model):
    """Get expected feature names from model."""
    if model is None:
        return None
    try:
        if hasattr(model, "feature_names_in_"):
            return list(model.feature_names_in_)
        if hasattr(model, "feature_names") and model.feature_names:
            return list(model.feature_names)
        if hasattr(model, "steps"):
            last = model.steps[-1][1]
            if hasattr(last, "feature_names_in_"):
                return list(last.feature_names_in_)
            if hasattr(last, "feature_names") and last.feature_names:
                return list(last.feature_names)
    except Exception as e:
        print(f"Warning: Could not introspect model features: {e}")
    return None

def parse_payload(required_columns, input_mapping=None):
    """Parse JSON payload with optional field mapping."""
    payload = request.get_json(silent=True)
    if payload is None:
        raise BadRequest("Invalid or missing JSON body. Ensure Content-Type: application/json and valid JSON.")
    
    rows = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(x, dict) for x in rows):
        raise BadRequest("Payload must be an object or an array of objects.")

    df = pd.DataFrame(rows)
    
    if input_mapping:
        df = df.rename(columns=input_mapping)
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            original_names = [k for k, v in input_mapping.items() if v in missing]
            raise BadRequest(f"Missing required fields: {original_names} (maps to {missing}). Provided keys: {list(payload.keys() if isinstance(payload, dict) else payload[0].keys())}")
    else:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise BadRequest(f"Missing required fields: {missing}. Provided keys: {list(df.columns)}")

    return df[required_columns]

def encode_structure_type(df):
    """Encode structure_type to StructureTypeEncoded."""
    if 'structure_type' in df.columns:
        df['StructureTypeEncoded'] = df['structure_type'].map(STRUCTURE_TYPE_ENCODING)
        unknown_mask = df['StructureTypeEncoded'].isna()
        if unknown_mask.any():
            unknown_types = df.loc[unknown_mask, 'structure_type'].unique().tolist()
            valid_types = list(STRUCTURE_TYPE_ENCODING.keys())
            raise BadRequest(f"Unknown structure_type values: {unknown_types}. Valid values: {valid_types}")
        df = df.drop('structure_type', axis=1)
    return df

def to_jsonable(preds):
    """Convert predictions to JSON-serializable format."""
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
    """Generate standardized error response."""
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
    return jsonify({
        "error": e.name,
        "message": e.description,
        "status_code": e.code,
    }), e.code

@app.errorhandler(Exception)
def handle_unexpected_exception(e: Exception):
    return error_response(e, 500)

# ==============================================================================
# HOME AND STATUS ENDPOINTS
# ==============================================================================

@app.route("/")
def home():
    # Show status of loaded models
    model_status = []
    models = [
        ("🌧️ Rainfall", rainfall_model, "/predict_rainfall_get?lat=26.83&lng=80.86"),
        ("💰 Cost", cost_model, "/predict_cost_get?structure_type=Pit&dwellers=14"),
        ("✅ Feasibility", feasibility_model, "/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83"),
        ("🏗️ Structure", structure_model, "/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4")
    ]
    
    for name, model, demo_url in models:
        status = "✅ Loaded" if model else "❌ Failed"
        if model:
            model_status.append(f'<li>{name}: {status} - <a href="{demo_url}">Test Demo</a></li>')
        else:
            model_status.append(f'<li>{name}: {status}</li>')
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rainwater Harvesting API</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .status {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .success {{ color: #28a745; }}
            .error {{ color: #dc3545; }}
            a {{ color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>🌧️ Rainwater Harvesting System API</h1>
        <p><strong>Developed by:</strong> karthi281104</p>
        <p><strong>Server Time:</strong> 2025-09-21 01:23:00 UTC</p>
        
        <div class="status">
            <h2>📊 Model Status:</h2>
            <ul>{''.join(model_status)}</ul>
        </div>
        
        <h2>🔗 Quick Links:</h2>
        <ul>
            <li><a href="/help">📖 Full Documentation</a></li>
            <li><a href="/models">🔧 Detailed Model Status</a></li>
            <li><a href="/healthz">❤️ Health Check</a></li>
        </ul>
        
        <h2>🧪 Working Demos:</h2>
        <p>Click these links to test the working models:</p>
        <ul>
    """

@app.route("/models")
def models_status():
    """Detailed model status and features."""
    def status(model, err):
        return "loaded" if model else f"error: {err}"
    
    return jsonify({
        "server_info": {
            "timestamp": "2025-09-21 01:23:00 UTC",
            "developer": "karthi281104",
            "python_version": "3.12",
            "flask_debug": DEBUG_ERRORS
        },
        "models": {
            "rainfall_model": {
                "status": status(rainfall_model, rainfall_err),
                "expected_features": get_expected_features_from_model(rainfall_model),
                "endpoint": "/predict_rainfall_get?lat=26.83&lng=80.86"
            },
            "cost_model": {
                "status": status(cost_model, cost_err),
                "expected_features": get_expected_features_from_model(cost_model),
                "endpoint": "/predict_cost_get?structure_type=Pit&dwellers=14"
            },
            "feasibility_model": {
                "status": status(feasibility_model, feasibility_err),
                "expected_features": get_expected_features_from_model(feasibility_model),
                "endpoint": "/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83"
            },
            "structure_model": {
                "status": status(structure_model, structure_err),
                "expected_features": get_expected_features_from_model(structure_model),
                "endpoint": "/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4"
            }
        },
        "structure_type_encoding": STRUCTURE_TYPE_ENCODING
    }), 200

@app.route("/healthz")
def health():
    working_models = sum(1 for model in [rainfall_model, cost_model, feasibility_model, structure_model] if model)
    return jsonify({
        "status": "ok",
        "timestamp": "2025-09-21 01:23:00 UTC",
        "models_loaded": f"{working_models}/4",
        "developer": "karthi281104"
    }), 200

# ==============================================================================
# GET ENDPOINTS (Browser-Friendly)
# ==============================================================================

@app.route("/predict_rainfall_get", methods=["GET"])
def predict_rainfall_get():
    """Rainfall prediction via GET request."""
    try:
        if not rainfall_model:
            return jsonify({
                "error": "Rainfall model not available",
                "details": rainfall_err,
                "suggestion": "Try other working endpoints like cost estimation"
            }), 503
        
        lat = request.args.get('lat', type=float)
        lng = request.args.get('lng', type=float)
        
        if lat is None or lng is None:
            return jsonify({
                "error": "Missing parameters",
                "required": ["lat", "lng"],
                "example": "/predict_rainfall_get?lat=26.83&lng=80.86"
            }), 400
        
        df = pd.DataFrame({'location_lat': [lat], 'location_lng': [lng]})
        preds = to_jsonable(rainfall_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        
        return jsonify({
            "rainfall_prediction": preds,
            "input": {"lat": lat, "lng": lng},
            "model": "rainfall_model",
            "status": "success"
        })
    except Exception as e:
        return error_response(e, 400)

@app.route("/predict_cost_get", methods=["GET"])
def predict_cost_get():
    """Cost estimation via GET request."""
    try:
        if not cost_model:
            return jsonify({
                "error": "Cost model not available",
                "details": cost_err
            }), 503
        
        structure_type = request.args.get('structure_type', type=str)
        dwellers = request.args.get('dwellers', type=int)
        
        if structure_type is None or dwellers is None:
            return jsonify({
                "error": "Missing parameters",
                "required": ["structure_type", "dwellers"],
                "example": "/predict_cost_get?structure_type=Pit&dwellers=14",
                "valid_structure_types": list(STRUCTURE_TYPE_ENCODING.keys())
            }), 400
        
        if structure_type not in STRUCTURE_TYPE_ENCODING:
            return jsonify({
                "error": "Invalid structure_type",
                "provided": structure_type,
                "valid_values": list(STRUCTURE_TYPE_ENCODING.keys())
            }), 400
        
        encoded_type = STRUCTURE_TYPE_ENCODING[structure_type]
        df = pd.DataFrame({
            'StructureTypeEncoded': [encoded_type],
            'dwellers': [dwellers]
        })
        
        preds = to_jsonable(cost_model.predict(df))
        if isinstance(preds, list) and len(preds) == 1:
            preds = preds[0]
        
        return jsonify({
            "cost_estimation_prediction": preds,
            "input": {
                "structure_type": structure_type,
                "dwellers": dwellers,
                "encoded_type": encoded_type
            },
            "model": "cost_model",
            "status": "success"
        })
    except Exception as e:
        return error_response(e, 400)

@app.route("/predict_feasibility_get", methods=["GET"])
def predict_feasibility_get():
    """Feasibility analysis via GET request."""
    try:
        if not feasibility_model:
            return jsonify({
                "error": "Feasibility model not available",
                "details": feasibility_err
            }), 503
        
        rainfall = request.args.get('rainfall', type=float)
        roof_area = request.args.get('roof_area', type=float)
        open_space = request.args.get('open_space', type=float)
        
        if any(x is None for x in [rainfall, roof_area, open_space]):
            return jsonify({
                "error": "Missing parameters",
                "required": ["rainfall", "roof_area", "open_space"],
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
            "input": {
                "rainfall": rainfall,
                "roof_area": roof_area,
                "open_space": open_space
            },
            "model": "feasibility_model",
            "status": "success"
        })
    except Exception as e:
        return error_response(e, 400)

@app.route("/predict_structure_get", methods=["GET"])
def predict_structure_get():
    """Structure type recommendation via GET request."""
    try:
        if not structure_model:
            return jsonify({
                "error": "Structure model not available",
                "details": structure_err
            }), 503
        
        annual_rainfall = request.args.get('annual_rainfall', type=float)
        roof_area = request.args.get('roof_area', type=float)
        open_space = request.args.get('open_space', type=float)
        
        if any(x is None for x in [annual_rainfall, roof_area, open_space]):
            return jsonify({
                "error": "Missing parameters",
                "required": ["annual_rainfall", "roof_area", "open_space"],
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
            "input": {
                "annual_rainfall": annual_rainfall,
                "roof_area": roof_area,
                "open_space": open_space
            },
            "model": "structure_model",
            "status": "success"
        })
    except Exception as e:
        return error_response(e, 400)

@app.route("/help")
def help_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { background: #007bff; color: white; padding: 3px 8px; border-radius: 3px; font-size: 12px; }
            .url { background: #e9ecef; padding: 8px; border-radius: 3px; font-family: monospace; }
            .working { border-left: 4px solid #28a745; }
            .broken { border-left: 4px solid #dc3545; }
            a { color: #007bff; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <h1>🌧️ Rainwater Harvesting System API</h1>
        <p><strong>Developer:</strong> karthi281104 | <strong>Updated:</strong> 2025-09-21</p>
        
        <h2>🟢 Working Endpoints (Click to Test):</h2>
        
        <div class="endpoint working">
            <h3>💰 Cost Estimation</h3>
            <span class="method">GET</span>
            <div class="url">/predict_cost_get?structure_type=Pit&dwellers=14</div>
            <p><a href="/predict_cost_get?structure_type=Pit&dwellers=14">🧪 Test Demo</a></p>
        </div>
        
        <div class="endpoint working">
            <h3>✅ Feasibility Analysis</h3>
            <span class="method">GET</span>
            <div class="url">/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83</div>
            <p><a href="/predict_feasibility_get?rainfall=1326&roof_area=71&open_space=83">🧪 Test Demo</a></p>
        </div>
        
        <div class="endpoint working">
            <h3>🏗️ Structure Type Recommendation</h3>
            <span class="method">GET</span>
            <div class="url">/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4</div>
            <p><a href="/predict_structure_get?annual_rainfall=336&roof_area=15&open_space=4">🧪 Test Demo</a></p>
        </div>
        
        <h2>🔴 Currently Unavailable:</h2>
        <div class="endpoint broken">
            <h3>🌧️ Rainfall Prediction</h3>
            <p><strong>Status:</strong> Model file corrupted/incompatible</p>
            <p><strong>Fix:</strong> Re-export the rainfall model with joblib.dump()</p>
        </div>
        
        <h2>🔧 Utility Endpoints:</h2>
        <ul>
            <li><a href="/models">📊 Detailed Model Status</a></li>
            <li><a href="/healthz">❤️ Health Check</a></li>
        </ul>
    </body>
    </html>
    """

# Add POST endpoints for the working models
@app.route("/predict_cost", methods=["POST"])
def predict_cost():
    """Cost estimation via POST request."""
    try:
        if not cost_model:
            return jsonify({"error": "Cost model not available", "details": cost_err}), 503
        
        payload = request.get_json(silent=True)
        if payload is None:
            raise BadRequest("Invalid or missing JSON body.")
        
        rows = payload if isinstance(payload, list) else [payload]
        df = pd.DataFrame(rows)
        
        required_input = ["structure_type", "dwellers"]
        missing = [c for c in required_input if c not in df.columns]
        if missing:
            raise BadRequest(f"Missing required fields: {missing}")
        
        df = encode_structure_type(df)
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
    """Feasibility analysis via POST request."""
    try:
        if not feasibility_model:
            return jsonify({"error": "Feasibility model not available", "details": feasibility_err}), 503
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
    """Structure type recommendation via POST request."""
    try:
        if not structure_model:
            return jsonify({"error": "Structure model not available", "details": structure_err}), 503
        df = parse_payload(STRUCTURE_FEATURES)
        preds = to_jsonable(structure_model.predict(df))
        if len(df) == 1 and isinstance(preds, list):
            preds = preds[0]
        return jsonify({"structure_type_prediction": preds})
    except HTTPException as e:
        raise e
    except Exception as e:
        return error_response(e, 400)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "1") == "1"
    
    print("="*60)
    print("🌧️ Rainwater Harvesting System API")
    print(f"👨‍💻 Developer: karthi281104")
    print(f"📅 Date: 2025-09-21")
    print("="*60)
    
    working_models = sum(1 for model in [rainfall_model, cost_model, feasibility_model, structure_model] if model)
    print(f"📊 Models Status: {working_models}/4 loaded successfully")
    
    if rainfall_model is None:
        print("⚠️  WARNING: Rainfall model failed to load")
        print("   💡 Suggestion: Re-export with joblib.dump() instead of pickle")
    
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📖 Help: http://{host}:{port}/help")
    print("="*60)
    
    app.run(host=host, port=port, debug=debug)
