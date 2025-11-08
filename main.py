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
    """
    Preprocesses a PIL Image for model prediction.
    - Resizes to (224, 224)
    - Converts to RGB (handles grayscale or 4-channel images)
    - Converts to NumPy array
    - Adds a batch dimension
    - *Does NOT normalize* (EfficientNetB0 does this internally)
    """
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
        st.warning("Model could not be loaded. Please check the model file.")
        return

    # Layout: reserve a place for the MRI image (uploaded or sample)
    st.subheader("Brain MRI Scan")
    image_container = st.empty()

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
        base_dir = os.path.dirname(__file__)

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
            image_container.image(image, caption="Uploaded Image", use_column_width=True)
            
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

            # 5. Display the results
            st.success(f"**Prediction:** {class_name}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            # # 6. (Optional) Show all probabilities
            # st.subheader("All Probabilities:")
            # # Create a dictionary for easier display
            # probs_dict = {CLASS_NAMES[i]: score[i] for i in range(len(CLASS_NAMES))}
            # st.bar_chart(probs_dict)

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        # No upload: show a sample MRI from the Testing folder if available
        sample_path = get_sample_image()
        if sample_path:
            try:
                sample_img = Image.open(sample_path)
                image_container.image(sample_img, caption=f"Sample Image — {os.path.basename(sample_path)}", use_column_width=True)
                st.info("This is a sample MRI image. Upload your own to classify.")
            except Exception as e:
                st.warning(f"Found sample image but couldn't open it: {e}")
        else:
            image_container.info("No sample MRI image found. Upload an image to begin.")

if __name__ == "__main__":
    main()
