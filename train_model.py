# xray_ai_training.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16 # Import a pre-trained model for transfer learning
import numpy as np
import os
import cv2 # OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt # For plotting training history

# --- Configuration ---
# Path to your main data directory containing 'normal' and 'abnormal' subfolders
DATA_DIR = '/Users/egaharshavardhan/xray_classification_data' # <<< UPDATE THIS PATH
# Path to save your trained model
MODEL_SAVE_PATH = 'xray_classification_model_transfer_learning.h5' # Changed model name
# Image dimensions for your model (VGG16 expects 224x224 or larger, but 256x256 is fine)
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3 # VGG16 expects 3 color channels (RGB), even for grayscale images. We'll convert.

# --- Data Loading and Preprocessing ---
def load_classification_data(data_dir, img_width, img_height, img_channels):
    """
    Loads X-ray images from 'normal' and 'abnormal' subfolders and assigns labels.
    Converts grayscale images to 3 channels for transfer learning models like VGG16.
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir)) # Should be ['abnormal', 'normal']

    print(f"Loading data from: {data_dir}")
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            print(f"Skipping non-directory: {class_path}")
            continue

        label = 1 if class_name == 'abnormal' else 0 # Assign 1 for abnormal, 0 for normal
        print(f"Processing class: {class_name} (label: {label})")

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale

                if img is None:
                    print(f"Warning: Could not read image {img_path}. Skipping.")
                    continue

                img = cv2.resize(img, (img_width, img_height))
                
                # Convert grayscale to 3 channels by stacking (required for VGG16)
                if img_channels == 3 and len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=-1)
                elif img_channels == 1 and len(img.shape) == 3: # If somehow 3 channel but should be 1
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


                img = img / 255.0 # Normalize pixel values to 0-1

                images.append(img)
                labels.append(label)
    
    images = np.array(images).reshape(-1, img_width, img_height, img_channels)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images.")
    if len(images) > 0:
        print(f"Image shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Label counts: Normal={np.sum(labels == 0)}, Abnormal={np.sum(labels == 1)}")
    return images, labels

# --- Model Architecture (Transfer Learning with VGG16) ---
def transfer_learning_model(input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
    # Load the VGG16 model pre-trained on ImageNet data
    # include_top=False means we don't include the classifier layers at the top
    # weights='imagenet' means use the pre-trained weights
    # input_shape specifies the input dimensions for our images
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_size)

    # Freeze the layers of the base model
    # This means their weights will not be updated during training
    for layer in base_model.layers:
        layer.trainable = False

    # Create the custom classification head
    x = base_model.output
    x = Flatten()(x) # Flatten the output from the convolutional base
    x = Dense(256, activation='relu')(x) # Add a fully connected layer
    x = Dropout(0.5)(x) # Add dropout for regularization
    predictions = Dense(1, activation='sigmoid')(x) # Output layer for binary classification

    # Combine the base model and the custom head
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# --- Main Training Script ---
if __name__ == '__main__':
    # 1. Load Data
    X, y = load_classification_data(DATA_DIR, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

    if X.size == 0:
        print("Exiting training: No data loaded. Please check DATA_DIR and image files.")
    else:
        # 2. Split Data (e.g., 80/20 train/validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y # Stratify to maintain class balance
        )

        print(f"Training images: {X_train.shape}, Training labels: {y_train.shape}")
        print(f"Validation images: {X_val.shape}, Validation labels: {y_val.shape}")
        print(f"Training label counts: Normal={np.sum(y_train == 0)}, Abnormal={np.sum(y_train == 1)}")
        print(f"Validation label counts: Normal={np.sum(y_val == 0)}, Abnormal={np.sum(y_val == 1)}")

        # 3. Data Augmentation (Crucial for small datasets)
        # Apply augmentation only to the training data
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            # horizontal_flip=True, # X-rays are usually not flipped horizontally due to anatomical left/right
            fill_mode='nearest'
        )

        # 4. Build and Compile Model (using transfer learning model)
        model = transfer_learning_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()

        # 5. Train Model
        print("\nStarting model training...")
        # You will need a GPU for efficient training.
        batch_size = 8 # Small batch size for small dataset
        epochs = 50 # Start with a reasonable number, use callbacks for early stopping
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
            ]
        )
        print("\nModel training complete.")

        # 6. Evaluate Model
        print("\nEvaluating model on validation data...")
        loss, accuracy = model.evaluate(X_val, y_val)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # Generate classification report and confusion matrix
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int) # Convert probabilities to binary predictions
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Normal', 'Abnormal']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_val, y_pred))

        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.show() # This will display the plots

        # 7. Save Model (already done by ModelCheckpoint callback)
        print(f"Best model saved to {MODEL_SAVE_PATH}")

        # --- Next Steps ---
        # 1. Integrate the saved model (xray_classification_model_transfer_learning.h5) into your Flask backend.
        # 2. Modify `run_ai_analysis` in Flask to load and use this model for actual inference.
        # 3. Deploy your Flask backend and connect your React frontend to it.
