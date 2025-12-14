import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_DIR = "COVID-19_Radiography_Dataset"  # Make sure this folder name matches exactly
IMG_SIZE = 128   # 128 is faster than 256 for a laptop
BATCH_SIZE = 16
EPOCHS = 5       # 5 Epochs is enough for a college demo

def load_data(data_dir, limit=800):
    images = []
    masks = []
    
    # Paths to images and masks
    img_path = os.path.join(data_dir, "Viral Pneumonia", "images")
    mask_path = os.path.join(data_dir, "Viral Pneumonia", "masks")
    
    print(f"Looking for data in: {img_path}")
    
    if not os.path.exists(img_path):
        print("ERROR: Folder not found! Check your folder structure.")
        return np.array([]), np.array([])

    files = os.listdir(img_path)[:limit] 
    
    for f in files:
        try:
            # Load & Resize Image (Grayscale)
            img = cv2.imread(os.path.join(img_path, f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1) 
            
            # Load & Resize Mask
            mask = cv2.imread(os.path.join(mask_path, f), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)
            
            images.append(img)
            masks.append(mask)
        except Exception as e:
            pass
            
    return np.array(images), np.array(masks)

# 1. Load Data
print("Loading dataset...")
X, Y = load_data(DATA_DIR)
print(f"Loaded {len(X)} images. Splitting data...")

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

# 2. Build U-Net Model
def build_unet(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u4 = UpSampling2D((2, 2))(c3)
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u4)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(32, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return Model(inputs=[inputs], outputs=[outputs])

print("Compiling Model...")
model = build_unet((IMG_SIZE, IMG_SIZE, 1))
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])

# 3. Train
print("Starting Training...")
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_val, Y_val))

# 4. Save
print("Saving model...")
if not os.path.exists("model_weights"):
    os.makedirs("model_weights")
    
model.save("model_weights/unet_pneumonia.h5")
print("SUCCESS! Model saved to model_weights/unet_pneumonia.h5")