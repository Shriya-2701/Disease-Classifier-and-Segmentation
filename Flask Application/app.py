import os
import numpy as np
from PIL import Image
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Import helper
from gradcam import make_gradcam_heatmap

app = Flask(__name__)

# ==========================================
# 1. LOAD MODELS
# ==========================================

# --- Model A: Classification & Infection (VGG19) ---
WEIGHTS_PATH_CLS = os.path.join(os.path.dirname(__file__), "..", "model_weights", "vgg19_model_03.h5")
base_model = VGG19(include_top=False, input_shape=(128, 128, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_cls = Model(base_model.inputs, output)

if os.path.exists(WEIGHTS_PATH_CLS):
    model_cls.load_weights(WEIGHTS_PATH_CLS)
    print("✅ VGG19 Loaded (Classification + Infection).")
else:
    print("❌ Error: VGG19 weights missing.")

# --- Model B: Lung Segmentation (U-Net) ---
WEIGHTS_PATH_SEG = os.path.join(os.path.dirname(__file__), "..", "model_weights", "unet_pneumonia.h5")
if os.path.exists(WEIGHTS_PATH_SEG):
    model_seg = load_model(WEIGHTS_PATH_SEG)
    print("✅ U-Net Loaded (Lung Segmentation).")
else:
    model_seg = None
    print("⚠️ U-Net weights missing.")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def preprocess_rgb(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

def preprocess_gray(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)

# --- FIXED LOGIC HERE ---
# Previous: 1=Normal.  
# Corrected (Alphabetical): 0=Normal, 1=Pneumonia.
def get_className(classNo):
    return "Pneumonia" if classNo == 1 else "Normal"

# ==========================================
# 3. PAGE ROUTES
# ==========================================

@app.route('/')
def home(): return render_template('home.html')

@app.route('/classification_page')
def classification_page(): return render_template('classify.html')

@app.route('/lung_seg_page')
def lung_seg_page(): return render_template('lung_seg.html') # Page 2

@app.route('/infection_seg_page')
def infection_seg_page(): return render_template('infection_seg.html') # Page 3


# ==========================================
# 4. API ENDPOINTS
# ==========================================

# --- API 1: CLASSIFICATION (VGG19) ---
@app.route('/predict_class', methods=['POST'])
def predict_class():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
    f.save(file_path)

    input_img = preprocess_rgb(file_path)
    result = model_cls.predict(input_img)
    class_idx = int(np.argmax(result, axis=1)[0])
    
    # Returns Corrected Class Name
    return jsonify({"result": get_className(class_idx)})


# --- API 2: LUNG SEGMENTATION (U-Net) ---
# Highlights the healthy/air-filled parts (Black regions)
@app.route('/predict_lung', methods=['POST'])
def predict_lung():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
    f.save(file_path)

    if model_seg is None:
        return jsonify({"status": "error", "message": "U-Net not loaded"})

    # U-Net needs Grayscale
    input_img = preprocess_gray(file_path)
    pred_mask = model_seg.predict(input_img)
    
    # Process Mask
    pred_mask = pred_mask[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # Draw Blue Lines for Lungs
    original_img = cv2.imread(file_path)
    original_img = cv2.resize(original_img, (128, 128))
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (255, 0, 0), 2) # Blue color for Organs

    result_filename = "lung_" + secure_filename(f.filename)
    result_path = os.path.join(basepath, 'static', 'uploads', result_filename)
    cv2.imwrite(result_path, original_img)

    return jsonify({"image_url": f"/static/uploads/{result_filename}"})


# --- API 3: INFECTION SEGMENTATION (VGG19 + Grad-CAM) ---
# Highlights the Pneumonia clouds (White regions)
@app.route('/predict_infection', methods=['POST'])
def predict_infection():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
    f.save(file_path)

    # VGG19 needs RGB
    input_img = preprocess_rgb(file_path)
    
    # Generate Heatmap (Grad-CAM) targeting White features
    heatmap = make_gradcam_heatmap(input_img, model_cls, 'block5_conv4')
    
    # Process Heatmap
    heatmap_resized = cv2.resize(heatmap, (128, 128))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    _, thresh = cv2.threshold(heatmap_uint8, 100, 255, cv2.THRESH_BINARY)
    
    # Draw Green Lines for Infection
    original_img = cv2.imread(file_path)
    original_img = cv2.resize(original_img, (128, 128))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(original_img, contours, -1, (0, 255, 0), 2) # Green color for Disease

    result_filename = "inf_" + secure_filename(f.filename)
    result_path = os.path.join(basepath, 'static', 'uploads', result_filename)
    cv2.imwrite(result_path, original_img)

    return jsonify({"image_url": f"/static/uploads/{result_filename}"})

if __name__ == '__main__':
    app.run(debug=True)