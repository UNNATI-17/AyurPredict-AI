from flask import Flask, render_template, request, jsonify
import os
import time
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Needed for unpickling
from model.script import AyurPredictSystem

warnings.filterwarnings("ignore")

# ----------------------------------------------------
# Flask App Initialization
# ----------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------
# Load Model, Encoders, and Data via AyurPredictSystem
# ----------------------------------------------------
BASE = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(BASE, "ayurpredict_trustworthy_optimized_model.pkl")
ENCODER_PATH = os.path.join(BASE, "feature_encoders.pkl")
DATA_PATH = os.path.join(BASE, "ayurpredict_model_ready.csv")

try:
    print("🔄 Initializing AyurPredict system...")
    system = AyurPredictSystem(MODEL_PATH, ENCODER_PATH, DATA_PATH)
    print("✅ AyurPredict system initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize AyurPredict system: {e}")
    system = None

# ----------------------------------------------------
# Web Page Routes
# ----------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/researcher')
def researcher():
    return render_template('researcher.html')

# ----------------------------------------------------
# API Routes
# ----------------------------------------------------

# 🔹 1️⃣ Symptom → Herbs (User Pathway)
@app.route('/api/analyze-symptoms', methods=['POST'])
def analyze_symptoms():
    if system is None:
        return jsonify({'error': 'Model not initialized properly.'}), 500

    data = request.get_json()
    symptom = data.get('symptoms', '').strip().lower()

    if not symptom:
        return jsonify({'error': 'No symptom provided.'}), 400

    try:
        start_time = time.time()
        result = system.symptom_pathway(symptom)
        elapsed = round(time.time() - start_time, 2)
        print(f"✅ Symptom '{symptom}' processed in {elapsed}s")

        if not result:
            return jsonify({'error': f"No results found for symptom '{symptom}'"}), 404

        # Include timing info in the response
        result['processing_time'] = elapsed
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error analyzing symptom: {e}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


# 🔹 2️⃣ Herb → Targets (Researcher Pathway)
@app.route('/api/analyze-herb', methods=['POST'])
def analyze_herb():
    if system is None:
        return jsonify({'error': 'Model not initialized properly.'}), 500

    data = request.get_json()
    herb = data.get('herb', '').strip().title()

    if not herb:
        return jsonify({'error': 'No herb provided.'}), 400

    try:
        start_time = time.time()
        result = system.researcher_pathway(herb)
        elapsed = round(time.time() - start_time, 2)
        print(f"✅ Herb '{herb}' processed in {elapsed}s")

        if not result:
            return jsonify({'error': f"No data found for herb '{herb}'"}), 404

        result['processing_time'] = elapsed
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error analyzing herb: {e}")
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


# ----------------------------------------------------
# Run Flask App
# ----------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
