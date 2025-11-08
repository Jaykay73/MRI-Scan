import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import glob

# --- 1. Constants and Model Loading ---

# Define the image size the model expects
IMG_SIZE = (224, 224)

# Update this path if your model file is named differently
MODEL_PATH = "efficientnet_best_model (1).keras"

# Define the class names in the correct alphabetical order
# This MUST match the order from training (e.g., from train_ds.class_names)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
# Friendly display names for predictions
DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary Tumor'
}

@st.cache_resource
def load_keras_model(model_path):
    """Loads the pre-trained Keras model."""
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 2. Image Preprocessing Function ---

def preprocess_image(image: Image.Image) -> np.ndarray:
  
    # 1. Convert to RGB
    image = image.convert('RGB')
    
    # 2. Resize the image
    image = image.resize(IMG_SIZE)
    
    # 3. Convert image to numpy array (pixels are 0-255)
    image_array = np.array(image)
    
    # 4. Add a batch dimension (e.g., (224, 224, 3) -> (1, 224, 224, 3))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# --- 3. Streamlit App Interface ---

def main():
    st.title("🧠 Brain Tumor MRI Classifier")
    st.markdown("Upload an MRI scan, and the model will predict the tumor type.")
    
    # Load the model
    model = load_keras_model(MODEL_PATH)
    
    if model is None:
        # Use error instead of warning when model cannot be loaded
        st.error("Model could not be loaded. Please check the model file.")
        return

    # Layout: reserve a place for the MRI image (uploaded or sample)
    st.subheader("Brain MRI Scan")
    # two-column layout: left for image, right for prediction/results
    left_col, right_col = st.columns([1, 1])
    image_container = left_col.empty()

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an MRI image...", 
        type=["jpg", "jpeg", "png"]
    )

    def get_sample_image():
        """Return the image path for Te-gl_0010.jpg if present, else the first image path
        found under the Testing/ folder, or None.
        Priority order:
        1. `Te-gl_0010.jpg` in the project root (next to main.py)
        2. Any image under the `Testing/` folder (first match)
        3. Any file named `Te-gl_0010.jpg` anywhere under the project tree
        """
        project_root = os.path.dirname(__file__)
        preferred = os.path.join(project_root, "Te-gl_0010.jpg")
        if os.path.isfile(preferred):
            return preferred

        # 2) Fallback: look for common image extensions in the Testing directory
        base = os.path.join(project_root, "Testing")
        if not os.path.isdir(base):
            return None
        patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
        for pat in patterns:
            matches = glob.glob(os.path.join(base, pat), recursive=True)
            if matches:
                return matches[0]
        return None

        # 1) Check for the specific filename next to main.py
        candidate_root = os.path.join(base_dir, "Te-gl_0010.jpg")
        if os.path.isfile(candidate_root):
            return candidate_root

        # 2) Look in the Testing/ directory for any common image
        testing_dir = os.path.join(base_dir, "Testing")
        if os.path.isdir(testing_dir):
            patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
            for pat in patterns:
                matches = glob.glob(os.path.join(testing_dir, pat), recursive=True)
                if matches:
                    return matches[0]

        # 3) As a final fallback, walk the project tree looking for Te-gl_0010.jpg
        for root, dirs, files in os.walk(base_dir):
            if "Te-gl_0010.jpg" in files:
                return os.path.join(root, "Te-gl_0010.jpg")

        return None

    if uploaded_file is not None:
        try:
            # 1. Load the image with PIL
            image = Image.open(uploaded_file)
            
            # 2. Display the uploaded image in the reserved container
            image_container.image(image, caption="Uploaded Image", use_column_width=False, width=360)
            
            # 3. Preprocess the image
            preprocessed_img = preprocess_image(image)
            
            # 4. Make prediction
            with st.spinner("Classifying..."):
                predictions = model.predict(preprocessed_img)
                
                # Get the score for the top prediction
                # predictions[0] is the array of probabilities for the single image
                score = predictions[0]
                class_index = np.argmax(score)
                class_name = CLASS_NAMES[class_index]
                confidence = np.max(score)
                
            # 5. Display the results in the right column (use friendly display name)
            display_name = DISPLAY_NAMES.get(class_name, class_name)
            right_col.success(f"**Prediction:** {display_name}")
            right_col.write(f"**Confidence:** {confidence:.2%}")
            

        except Exception as e:
            right_col.error(f"An error occurred: {e}")
    else:
        # No upload: show a sample MRI from the Testing folder if available
        sample_path = get_sample_image()
        if sample_path:
            try:
                sample_img = Image.open(sample_path)
                image_container.image(sample_img, caption=f"Sample Image — {os.path.basename(sample_path)}", use_column_width=False, width=360)
                right_col.info("This is a sample MRI image. Upload your own to classify.")
            except Exception as e:
                # replace warning with info to avoid showing warning UI
                right_col.info(f"Found sample image but couldn't open it: {e}")
        else:
            image_container.info("No sample MRI image found. Upload an image to begin.")
            right_col.info("Place 'Te-gl_0010.jpg' next to main.py or upload an image to classify.")

if __name__ == "__main__":
    main()
