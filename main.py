# ============================================
# Plant Disease Detection API
# Flask + TFLite + Supabase
# Deploy on: Render.com (FREE)
# ============================================

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
import os
import json
from supabase import create_client, Client

app = Flask(__name__)

# ---- Supabase Config ----
SUPABASE_URL = "https://cmhnzzhpoizibkmdbuvz.supabase.co"
SUPABASE_KEY = "sb_publishable_rOip2vRgkOF6TKE81yHtQA_4l-z-S8G"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- Labels ----
LABELS = [
    "Apple Brown_spot",
    "Apple Normal",
    "Apple black_spot",
    "Apricot Normal",
    "Apricot blight leaf disease",
    "Apricot shot_hole",
    "Bean Fungal_leaf disease",
    "Bean Normal leaf",
    "Bean bean rust image",
    "Bean shot_hole",
    "Cherry Leaf Scorch",
    "Cherry Normal leaf",
    "Cherry brown_spot",
    "Cherry purple leaf spot",
    "Cherry_shot hole disease",
    "Corn Fungal leaf",
    "Corn Normal leaf",
    "Corn gray leaf spot",
    "Corn holcus_ leaf spot",
    "Fig Blight_leaf disease",
    "Fig Brown spot",
    "Fig normal leaf",
    "Fig_rust leaf",
    "Grape Anthracnose leaf",
    "Grape Brown spot leaf",
    "Grape Downy mildew leaf",
    "Grape Mites_leaf disease",
    "Grape Normal_leaf",
    "Grape Powdery_mildew leaf",
    "Grape shot hole leaf disease",
    "Lokat Normal leaf",
    "Pear Black spot _ leaf disease",
    "Pear Normal _leaf",
    "Pear fire blight",
    "Walnut Anthracnose_leaf disease",
    "Walnut Blotch_leaf disease",
    "Walnut Normal_leaf",
    "Walnut Shot_hole",
    "Walnut leaf gall mite",
    "lokat Leaf_spot",
    "persimmons Brown_spot",
    "tomato Fusarium Wilt",
    "tomato spider mites",
    "tomato verticillium wilt",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy_leaf",
    "tomato_late_blight",
    "tomato_leaf_curl",
    "tomato_leaf_miner",
    "tomato_leaf_mold",
    "tomato_septoria_leaf"
]

# ---- Healthy labels ----
HEALTHY_LABELS = [
    "Apple Normal", "Apricot Normal", "Bean Normal leaf",
    "Cherry Normal leaf", "Corn Normal leaf", "Fig normal leaf",
    "Grape Normal_leaf", "Lokat Normal leaf", "Pear Normal _leaf",
    "Walnut Normal_leaf", "tomato_healthy_leaf"
]

# ---- Load Model ----
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="model/mobilenetv2_plantcity.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model loaded!")

# ========================================
# Image Preprocess
# ========================================
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================================
# Disease Detect
# ========================================
def detect_disease(image_bytes):
    img = preprocess_image(image_bytes)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    top_index      = int(np.argmax(predictions))
    confidence     = float(predictions[top_index]) * 100
    disease_label  = LABELS[top_index]
    is_healthy     = disease_label in HEALTHY_LABELS
    
    # Top 3 predictions
    top3_indices = np.argsort(predictions)[-3:][::-1]
    top3 = [
        {
            "label": LABELS[i],
            "confidence": round(float(predictions[i]) * 100, 2)
        }
        for i in top3_indices
    ]
    
    return {
        "disease":    disease_label,
        "confidence": round(confidence, 2),
        "is_healthy": is_healthy,
        "top3":       top3
    }

# ========================================
# Health Check
# ========================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status":  "running",
        "model":   "MobileNetV2 Plant Disease",
        "classes": len(LABELS)
    })

# ========================================
# Main Detection Endpoint
# ESP32-CAM ya Supabase webhook yahan aayega
# ========================================
@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        
        if not data or "image_url" not in data:
            return jsonify({"error": "image_url required"}), 400
        
        image_url = data["image_url"]
        scan_id   = data.get("scan_id", None)
        
        print(f"Detecting: {image_url}")
        
        # Image download karo
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({"error": "Image download failed"}), 400
        
        # Disease detect karo
        result = detect_disease(response.content)
        
        print(f"Result: {result['disease']} ({result['confidence']}%)")
        
        # Supabase mein update karo
        if scan_id:
            supabase.table("leaf_scans").update({
                "status":     "completed",
                "disease":    result["disease"],
                "confidence": result["confidence"],
                "is_healthy": result["is_healthy"],
                "top3":       json.dumps(result["top3"])
            }).eq("id", scan_id).execute()
            print(f"Supabase updated for scan_id: {scan_id}")
        
        return jsonify({
            "success":    True,
            "disease":    result["disease"],
            "confidence": result["confidence"],
            "is_healthy": result["is_healthy"],
            "top3":       result["top3"],
            "message":    "Healthy!" if result["is_healthy"] 
                          else f"Disease detected: {result['disease']}"
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ========================================
# Webhook — Supabase se auto trigger
# ========================================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        
        # Supabase webhook format
        record = data.get("record", {})
        scan_id   = record.get("id")
        image_url = record.get("image_url")
        status    = record.get("status")
        
        # Sirf pending records process karo
        if status != "pending":
            return jsonify({"message": "Skipped"}), 200
        
        if not image_url:
            return jsonify({"error": "No image_url"}), 400
        
        print(f"Webhook triggered for scan_id: {scan_id}")
        
        # Image download
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            return jsonify({"error": "Image download failed"}), 400
        
        # Detect
        result = detect_disease(response.content)
        
        # Supabase update
        supabase.table("leaf_scans").update({
            "status":     "completed",
            "disease":    result["disease"],
            "confidence": result["confidence"],
            "is_healthy": result["is_healthy"],
            "top3":       json.dumps(result["top3"])
        }).eq("id", scan_id).execute()
        
        print(f"Detection complete: {result['disease']} ({result['confidence']}%)")
        
        return jsonify({"success": True, "result": result})
        
    except Exception as e:
        print(f"Webhook error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
