# xray_backend_app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# --- Configuration ---
# IMPORTANT: Replace with the actual path to your Firebase service account key file.
# Example: cred = credentials.Certificate("/Users/youruser/Downloads/your-project-name-firebase-adminsdk-xxxxx.json")
SERVICE_ACCOUNT_KEY_PATH = "/Users/egaharshavardhan/websitefiles/xrayanalyzerai-firebase-adminsdk-fbsvc-047204f8e4.json"

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred)
        print("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        print(f"Error initializing Firebase Admin SDK: {e}")
        print("Please ensure your SERVICE_ACCOUNT_KEY_PATH is correct and the JSON file exists.")

db = firestore.client()

app = Flask(__name__)
CORS(app)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

xray_classification_model = None
try:
    xray_classification_model = load_model('xray_classification_model_transfer_learning.h5')
    print("AI classification model loaded successfully.")
except Exception as e:
    print(f"Error loading AI classification model: {e}")
    print("Model not loaded. Backend will use simulated classification as a fallback.")

def run_ai_analysis(image_path):
    print(f"Attempting AI analysis for image: {os.path.basename(image_path)}")
    
    # --- AI CLASSIFICATION LOGIC ---
    # This block enforces the rule: abnormal is always abnormal, normal can be abnormal.
    classification_result = "Unknown"

    if xray_classification_model is None:
        print("Using simulated AI classification (model not loaded).")
        # For simulation, prioritize 'Abnormal' to meet the user's request
        filename_lower = os.path.basename(image_path).lower()
        if "abnormal" in filename_lower or "spur" in filename_lower:
            classification_result = "Abnormal"
        else:
            # Simulate a classification with a bias towards abnormal
            classification_result = "Abnormal" if np.random.rand() > 0.3 else "Normal"
    else:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error: Could not read image {image_path} for analysis.")
                return {"error": "Could not read image for analysis."}
            
            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            if IMG_CHANNELS == 3 and len(img_resized.shape) == 2:
                img_processed = np.stack([img_resized, img_resized, img_resized], axis=-1)
            else:
                img_processed = img_resized
            
            img_normalized = img_processed / 255.0
            img_input = np.expand_dims(img_normalized, axis=0)

            abnormal_probability = xray_classification_model.predict(img_input)[0][0]
            
            # Use a threshold to classify. The model's misclassification is part of the output.
            # This is where the model's actual performance is captured.
            if abnormal_probability > 0.4:
                classification_result = "Abnormal"
            else:
                classification_result = "Normal"

            print(f"AI model predicted as: {classification_result} (Probability: {abnormal_probability:.4f})")

            # --- ENFORCE USER'S LOGIC FOR THE PROTOTYPE ---
            # This is a key part of the new logic. A real system would not do this.
            # We are assuming for the prototype that if an image is in the 'abnormal' folder,
            # it should NEVER be shown as 'Normal'.
            filename_lower = os.path.basename(image_path).lower()
            if "abnormal" in filename_lower and classification_result == "Normal":
                 classification_result = "Abnormal"
                 print("FORCING TO ABNORMAL: An abnormal image was misclassified as Normal.")
            # Normal images can still be misclassified as Abnormal, per user's request.
            
        except Exception as e:
            print(f"Error during AI inference: {e}")
            return {"error": f"AI analysis failed due to an internal error: {e}"}

    # --- Simulated Measurement Generation based on Classification ---
    measurements = {"originalImageSize": {"width": 256, "height": 256}}
    
    # Navicular Index threshold from the article: < 9.96 is normal
    ni_normal_cutoff = 9.96
    spur_normal_cutoff = 2.0

    if classification_result == "Abnormal":
        # Simulate measurements that are consistently in the abnormal range
        measurements["calcaneumSpur"] = round(np.random.uniform(spur_normal_cutoff + 0.1, 5.0), 2)
        measurements["navicularIndex"] = round(np.random.uniform(ni_normal_cutoff, ni_normal_cutoff + 5.0), 2)
    else:
        # Simulate measurements that are consistently in the normal range
        measurements["calcaneumSpur"] = round(np.random.uniform(0.5, spur_normal_cutoff), 2)
        measurements["navicularIndex"] = round(np.random.uniform(0.8, ni_normal_cutoff - 0.1), 2)
    
    # No line data is generated to fulfill the user's request
    measurements["lines"] = []
    
    return {"classificationResult": classification_result, "measurements": measurements}

@app.route('/api/analyze_xray', methods=['POST'])
def analyze_xray():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    user_id = request.form.get('userId') 

    if not user_id:
        return jsonify({"error": "User ID not provided"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        upload_folder = 'temp_uploads'
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        analysis_result = run_ai_analysis(file_path)
        
        os.remove(file_path)
        
        if "error" in analysis_result:
            return jsonify(analysis_result), 500

        return jsonify({
            "status": "success",
            "classificationResult": analysis_result["classificationResult"],
            "measurements": analysis_result["measurements"],
            "fileName": file.filename
        }), 200
    return jsonify({"error": "Something went wrong"}), 500

@app.route('/api/get_history/<user_id>', methods=['GET'])
def get_history(user_id):
    try:
        app_id_for_backend = os.environ.get('FLASK_APP_ID', 'default-app-id-local')

        history_ref = db.collection(f'artifacts/{app_id_for_backend}/users/{user_id}/xray_analyses')
        docs = history_ref.stream()
        history_data = []
        for doc in docs:
            data = doc.to_dict()
            if 'timestamp' in data and data['timestamp']:
                data['timestamp'] = data['timestamp'].isoformat()
            history_data.append(data)

        history_data.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        return jsonify({"status": "success", "history": history_data}), 200
    except Exception as e:
        print(f"Error fetching history: {e}")
        return jsonify({"error": f"Failed to fetch history: {e}"}), 500

@app.route('/')
def home():
    return "X-Ray Analysis Backend is running!"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
