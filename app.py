"""
=============================================================================
IPO Hype vs Reality — Flask Web Application (Step 12)
=============================================================================
This Flask application serves as the deployment layer for the IPO prediction
ML pipeline. It loads pre-trained model artifacts (.pkl files), accepts user
input through a web form, engineers the required features, and returns a
prediction with confidence scores.

HOW IT WORKS (end-to-end flow):
────────────────────────────────
1. The user fills out a form on the frontend (index.html) with IPO details
   such as issue price, subscription times, sector, sentiment, and news
   frequency.
2. When the user clicks "Predict IPO Outcome", the browser sends an HTTP
   POST request to the /predict route with the form data.
3. The backend (this file) receives the form data, constructs a feature
   vector that matches the training data schema, scales it using the saved
   scaler, and passes it through the saved ML model.
4. The model returns a class prediction (0, 1, or 2) and probability
   estimates. These are sent back to the frontend via Jinja2 template
   rendering, and the result is displayed to the user.

PKL FILES (model artifacts):
────────────────────────────
- best_model.pkl  : The trained classifier (e.g., Random Forest) saved via
                    joblib after hyperparameter tuning.
- scaler.pkl      : The StandardScaler / MinMaxScaler fitted on the training
                    data, used to normalize input features before prediction.
- feature_names.pkl: A list of feature column names in the exact order the
                     model expects, including one-hot encoded sector columns.
=============================================================================
"""

# ── Standard library & third-party imports ──────────────────────────────────
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import os

# ── Initialize the Flask application ───────────────────────────────────────
# Flask looks for templates in ./templates/ and static files in ./static/
app = Flask(__name__)

# ── Load pre-trained model artifacts ────────────────────────────────────────
# These .pkl files are generated during the training pipeline (Steps 8-11).
# joblib is used because it handles large numpy arrays inside sklearn models
# more efficiently than Python's built-in pickle module.

# Use absolute paths relative to app.py's directory for robust loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(BASE_DIR, "feature_names.pkl")

model = None
scaler = None
feature_names = None

def load_artifacts():
    global model, scaler, feature_names
    
    if model is None:
        try:
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                print(f"[SUCCESS] Model loaded successfully from '{MODEL_PATH}'")
            else:
                print(f"[ERROR] 'best_model.pkl' not found at '{MODEL_PATH}'!")
                print("   -> Make sure you have trained the model and saved it as 'best_model.pkl'")
        except Exception as e:
            print(f"[ERROR] loading model: {e}")
            
    if scaler is None:
        try:
            if os.path.exists(SCALER_PATH):
                scaler = joblib.load(SCALER_PATH)
                print(f"[SUCCESS] Scaler loaded successfully from '{SCALER_PATH}'")
            else:
                print(f"[ERROR] 'scaler.pkl' not found at '{SCALER_PATH}'!")
        except Exception as e:
            print(f"[ERROR] loading scaler: {e}")
            
    if feature_names is None:
        try:
            if os.path.exists(FEATURE_NAMES_PATH):
                feature_names = joblib.load(FEATURE_NAMES_PATH)
                print(f"[SUCCESS] Feature names loaded successfully ({len(feature_names)} features) from '{FEATURE_NAMES_PATH}'")
            else:
                print(f"[ERROR] 'feature_names.pkl' not found at '{FEATURE_NAMES_PATH}'!")
        except Exception as e:
            print(f"[ERROR] loading feature names: {e}")

# Try to load on startup
load_artifacts()

# ── Sector Popularity Mapping ──────────────────────────────────────────────
# Hardcoded from the training dataset's sector distribution / popularity
# analysis. Higher values indicate more popular sectors in the IPO market.
SECTOR_POPULARITY = {
    "Food Tech": 8,
    "Fintech": 9,
    "E-commerce": 10,
    "Insurance": 7,
    "Logistics": 7,
    "QSR": 6,
    "Automobile": 5,
    "Pharma": 6,
    "Gaming": 4,
    "Fashion": 5,
    "FMCG": 6,
    "Footwear": 3,
    "Drone Tech": 3,
    "Travel Tech": 5,
    "Infrastructure": 4,
    "Real Estate": 5,
    "Green Energy": 7,
    "IT": 8,
    "Microfinance": 4,
    "Industrial": 4,
    "Solar Energy": 6,
    "Construction": 4,
    "Finance": 7,
    "Retail": 6,
    "Auto Components": 3,
}

# ── All sector names (must match training data one-hot columns) ────────────
ALL_SECTORS = list(SECTOR_POPULARITY.keys())

# ── Prediction label mapping ───────────────────────────────────────────────
# Maps the model's numeric output to human-readable labels.
# The order must match the label encoding used during training.
PREDICTION_LABELS = {
    0: "Overhyped IPO",
    1: "Successful IPO",
    2: "Underperforming IPO",
}

# ── CSS class mapping for color coding on the frontend ─────────────────────
PREDICTION_CSS_CLASSES = {
    "Successful IPO": "success",
    "Overhyped IPO": "overhyped",
    "Underperforming IPO": "underperforming",
}


# ════════════════════════════════════════════════════════════════════════════
# ROUTE: Home Page
# ════════════════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    """
    Renders the main page with the prediction form.
    The template receives the list of sectors to populate the dropdown.
    No prediction is shown on initial load (prediction=None).
    """
    return render_template("index.html", sectors=ALL_SECTORS, prediction=None)


# ════════════════════════════════════════════════════════════════════════════
# ROUTE: Predict IPO Outcome
# ════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles the form submission from the frontend.

    Flow:
    ─────
    1. Extract form data sent via POST
    2. Engineer derived features (hype_score, sector_popularity, ipo_age_days)
    3. One-hot encode the sector
    4. Build a DataFrame matching the exact feature order from training
    5. Scale features using the saved scaler
    6. Run prediction through the saved model
    7. Return results to the template for display
    """
    
    # Try to load artifacts dynamically if they are still None (e.g. if created after start)
    load_artifacts()

    # ── Guard: check that all artifacts are loaded ─────────────────────────
    if model is None or scaler is None or feature_names is None:
        return render_template(
            "index.html",
            sectors=ALL_SECTORS,
            prediction="Error",
            confidence=0,
            css_class="underperforming",
            explanation="Model artifacts not loaded. Please ensure best_model.pkl, "
                        "scaler.pkl, and feature_names.pkl are in the project directory.",
        )

    try:
        # ── Step 1: Extract raw form inputs ────────────────────────────────
        # request.form is a dictionary-like object containing the POST data
        # sent by the HTML form. Each key matches the 'name' attribute of
        # the corresponding <input> or <select> element.
        issue_price = float(request.form["issue_price"])
        subscribed_times = float(request.form["subscribed_times"])
        sector = request.form["sector"]
        avg_sentiment = float(request.form["avg_sentiment"])
        news_frequency = int(request.form["news_frequency"])

        # ── Step 2: Compute derived features ───────────────────────────────
        # hype_score: A composite metric combining subscription demand,
        # media sentiment, and news coverage. Normalized to roughly 0-1 range.
        hype_score = (subscribed_times / 100 + avg_sentiment + news_frequency / 10) / 3

        # sector_popularity: Looked up from the hardcoded dictionary based
        # on historical IPO data analysis.
        sector_popularity = SECTOR_POPULARITY.get(sector, 5)  # default 5 if unknown

        # ipo_age_days: Set to 30 for a "new" IPO that's just been listed.
        # In the training data this represents days since listing.
        ipo_age_days = 30

        # ── Step 3: Build the feature dictionary ───────────────────────────
        # Start with the numeric features
        feature_dict = {
            "issue_price": issue_price,
            "subscribed_times": subscribed_times,
            "avg_sentiment": avg_sentiment,
            "news_frequency": news_frequency,
            "sector_popularity": sector_popularity,
            "hype_score": hype_score,
            "ipo_age_days": ipo_age_days,
        }

        # ── Step 4: One-hot encode the sector ─────────────────────────────
        # Create a binary column for every sector in the training data.
        # Only the selected sector gets a 1; all others are 0.
        # Column names must match exactly: "sector_<SectorName>"
        for s in ALL_SECTORS:
            col_name = f"sector_{s}"
            feature_dict[col_name] = 1 if s == sector else 0

        # ── Step 5: Build DataFrame in the correct column order ───────────
        # The model expects features in the EXACT same order as during
        # training. We use the saved feature_names list to reindex.
        input_df = pd.DataFrame([feature_dict])

        # Add any missing columns (in case training had extra features)
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match training order
        input_df = input_df[feature_names]

        # ── Step 6: Scale the features ─────────────────────────────────────
        # The scaler (StandardScaler/MinMaxScaler) was fit on the training
        # data. We transform the input to the same scale so the model
        # receives properly normalized values.
        input_scaled = scaler.transform(input_df)

        # ── Step 7: Make prediction ────────────────────────────────────────
        # predict() returns the class label (0, 1, or 2)
        prediction_num = model.predict(input_scaled)[0]

        # predict_proba() returns probability estimates for each class
        # We take the max probability as our confidence score.
        prediction_proba = model.predict_proba(input_scaled)[0]
        confidence = round(float(np.max(prediction_proba)) * 100, 2)

        # Map numeric prediction to human-readable label
        prediction_label = PREDICTION_LABELS.get(prediction_num, "Unknown")

        # Get the CSS class for color-coding the result
        css_class = PREDICTION_CSS_CLASSES.get(prediction_label, "")

        # ── Step 8: Generate explanation text ──────────────────────────────
        explanations = {
            "Successful IPO": (
                f"This IPO in the {sector} sector shows strong fundamentals with "
                f"{subscribed_times}x subscription and a hype score of {hype_score:.2f}. "
                f"The model predicts a successful listing with {confidence}% confidence."
            ),
            "Overhyped IPO": (
                f"Despite {subscribed_times}x subscription in the {sector} sector, "
                f"the sentiment ({avg_sentiment:.2f}) and hype indicators suggest this "
                f"IPO may be overvalued. Proceed with caution — {confidence}% confidence."
            ),
            "Underperforming IPO": (
                f"The {sector} sector IPO with {subscribed_times}x subscription and "
                f"sentiment of {avg_sentiment:.2f} shows signs of underperformance. "
                f"The model suggests limited upside — {confidence}% confidence."
            ),
        }
        explanation = explanations.get(prediction_label, "")

        # ── Step 9: Render template with results ──────────────────────────
        # The prediction, confidence, and explanation are passed to the
        # Jinja2 template engine, which injects them into index.html.
        return render_template(
            "index.html",
            sectors=ALL_SECTORS,
            prediction=prediction_label,
            confidence=confidence,
            css_class=css_class,
            explanation=explanation,
            # Pass back form values so the form retains user input
            form_data={
                "issue_price": issue_price,
                "subscribed_times": subscribed_times,
                "sector": sector,
                "avg_sentiment": avg_sentiment,
                "news_frequency": news_frequency,
            },
        )

    except ValueError as ve:
        # Handles cases where form inputs can't be converted to numbers
        return render_template(
            "index.html",
            sectors=ALL_SECTORS,
            prediction="Input Error",
            confidence=0,
            css_class="underperforming",
            explanation=f"Invalid input: {ve}. Please check your values and try again.",
        )
    except Exception as e:
        # Catch-all for unexpected errors
        return render_template(
            "index.html",
            sectors=ALL_SECTORS,
            prediction="Error",
            confidence=0,
            css_class="underperforming",
            explanation=f"An unexpected error occurred: {e}",
        )


# ════════════════════════════════════════════════════════════════════════════
# Run the Flask development server
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # debug=True enables:
    #   - Auto-reload when code changes
    #   - Detailed error pages in the browser
    # port=5000 is Flask's default; access at http://localhost:5000
    print("\n" + "=" * 60)
    print("  [SERVER] IPO Hype vs Reality - Prediction Server")
    print("  [INFO] Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
