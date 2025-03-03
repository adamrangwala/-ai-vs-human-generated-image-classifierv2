import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
import gc
import traceback
import psutil
import inspect

# Configure page
st.set_page_config(page_title="AI vs Human Image Classifier", layout="wide")

# Memory monitoring function
def get_memory_usage():
    """Returns the current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)

# Streamlit header
st.title("AI vs Human Image Classifier")
st.markdown("Upload a .keras or .h5 model file to classify images as AI-generated or human-generated.")

# Display current memory usage if in debug mode
if st.checkbox("Show memory usage", value=False):
    memory_usage = get_memory_usage()
    memory_container = st.empty()
    memory_container.info(f"Current memory usage: {memory_usage:.2f} MB")

# File size validator
def validate_file_size(file, max_size_mb=500):
    """Validates that the uploaded file isn't too large"""
    if file.size > max_size_mb * 1024 * 1024:
        st.error(f"File is too large! Maximum size is {max_size_mb}MB.")
        return False
    return True




# File uploader with size limit warning
st.warning("Note: Large model files (>100MB) may cause memory issues. Consider using a smaller or quantized model.")
uploaded_file = st.file_uploader("Choose a model file", type=["keras", "h5"])

def get_custom_objects_from_code(code_string):
    """Evaluates the custom objects code and returns a dictionary of custom objects"""
    custom_objects = {}
    try:
        # Create a local namespace
        local_namespace = {}
        
        # Add tensorflow to the namespace
        local_namespace['tf'] = tf
        
        # Execute the code in the local namespace
        exec(code_string, globals(), local_namespace)
        
        # Find all custom layer classes in the namespace
        for name, obj in local_namespace.items():
            if (inspect.isclass(obj) and 
                issubclass(obj, tf.keras.layers.Layer) and 
                obj != tf.keras.layers.Layer):
                custom_objects[name] = obj
                
        return custom_objects
    except Exception as e:
        st.error(f"Error in custom objects code: {str(e)}")
        st.code(traceback.format_exc(), language="python")
        return {}

@st.cache_resource(show_spinner=False)
def load_keras_model(model_path, custom_objects_dict=None):
    """Loads and caches the Keras model to prevent reloading on each interaction."""
    try:
        # Configure TensorFlow for memory efficiency
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                st.warning(f"GPU memory growth setting failed: {e}")
        
        # Set TensorFlow to log device placement if debugging
        tf.debugging.set_log_device_placement(False)
        
        # Try to load the model with CPU first for stability
        with tf.device('/CPU:0'):
            model = load_model(model_path, custom_objects=custom_objects_dict)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        return None

@st.cache_data
def preprocess_image(_image, target_size=(224, 224)):
    """Preprocess image for model input with caching."""
    try:
        # Convert to RGB if needed
        if _image.mode != "RGB":
            _image = _image.convert("RGB")
            
        # Resize with proper error handling
        try:
            _image = _image.resize(target_size, Image.LANCZOS)
        except (AttributeError, ValueError):
            # Fallback to NEAREST for problematic images
            _image = _image.resize(target_size, Image.NEAREST)
            
        # Convert to array and normalize to [0,1]
        img_array = np.array(_image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

@st.cache_data
def predict_image(model, img_array):
    """Make and cache model predictions."""
    if img_array is None:
        return None
        
    try:
        # Predictions on CPU for stability
        with tf.device('/CPU:0'):
            result = model.predict(img_array, verbose=0)
            
        # Handle different output shapes
        if isinstance(result, list):
            return result[0][0] if len(result) > 0 and len(result[0]) > 0 else 0.5
        elif result.ndim > 1:
            return result[0][0] if result.shape[0] > 0 and result.shape[1] > 0 else 0.5
        else:
            return result[0] if result.size > 0 else 0.5
            
    except (IndexError, AttributeError, ValueError) as e:
        st.warning(f"Prediction shape issue: {str(e)}. Model may have unexpected output format.")
        return 0.5
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Main application logic
if uploaded_file is not None:
    if not validate_file_size(uploaded_file, max_size_mb=500):
        st.stop()
        
    try:
        # Extract custom objects from code
        with st.spinner("Processing custom objects..."):
            custom_objects = get_custom_objects_from_code(custom_objects_code)
            if custom_objects:
                st.success(f"Found {len(custom_objects)} custom objects: {', '.join(custom_objects.keys())}")
            else:
                st.info("No custom objects found or defined. Using default model loading.")
        
        # Progress indicator for model loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Saving uploaded model to temporary file...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_filename = tmp_file.name
        
        progress_bar.progress(0.3)
        status_text.text("Loading model (this may take a while for large models)...")
        
        with st.spinner("Loading model..."):
            model = load_keras_model(tmp_filename)

        # Clean up temp file
        try:
            os.unlink(tmp_filename)
        except Exception as e:
            st.warning(f"Could not delete temporary file: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.empty()
        progress_bar.empty()
        
        if model is None:
            st.error("Failed to load model. Please check the model format and try again.")
            st.stop()
            
        st.success(f"Model loaded successfully from {uploaded_file.name}")

        # Force garbage collection
        gc.collect()

        # Get input shape from model
        try:
            input_shape = model.input_shape[1:3]
            if None in input_shape:
                input_shape = (224, 224)  # Default size if not specified
        except (IndexError, AttributeError):
            input_shape = (224, 224)  # Default fallback
                
        st.info(f"Model expects input images of size {input_shape}")

        # Image upload for prediction
        st.subheader("Upload an image to classify")
        uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
        
        # Columns for display
        col1, col2 = st.columns(2)
        
        if uploaded_img is not None:
            try:
                # Validate image size
                if not validate_file_size(uploaded_img, max_size_mb=10):
                    st.warning("Image is too large, please upload a smaller image.")
                else:
                    # Display image
                    image = Image.open(uploaded_img)
                    if image is None:
                        st.error("Failed to load image. Please try another file.")
                        st.stop()
                        
                    with col1:
                        st.image(image, caption="Uploaded Image", use_column_width=True)
                        
                        # Show image details
                        width, height = image.size
                        st.caption(f"Original dimensions: {width}x{height} pixels, {image.mode} mode")
                        st.caption(f"Resizing to {input_shape} for model input")
                    
                    with st.spinner("Analyzing image..."):
                        # Preprocess and predict
                        image_array = preprocess_image(image, target_size=input_shape)
                        
                        if image_array is not None:
                            probability = predict_image(model, image_array)
                            
                            if probability is not None:
                                with col2:
                                    confidence = round(probability * 100, 2) if probability > 0.5 else round((1 - probability) * 100, 2)
                                    label = "AI-Generated" if probability > 0.5 else "Human-Generated"
                                    color = "red" if probability > 0.5 else "green"
                                    emoji = "🤖" if probability > 0.5 else "🧑"

                                    # Create some vertical space
                                    for _ in range(3):
                                        st.write("")
                                    
                                    # Display prediction
                                    st.markdown(f"<h1 style='text-align: center; color: {color};'>{label}</h1>", unsafe_allow_html=True)
                                    st.markdown(f"<h2 style='text-align: center; color: black;'>{confidence}% confidence</h2>", unsafe_allow_html=True)
                                    st.markdown(f"<h1 style='text-align: center;'>{emoji}</h1>", unsafe_allow_html=True)
                                    
                                    # Display raw prediction
                                    st.caption(f"Raw prediction value: {probability:.4f}")
                            else:
                                st.error("Failed to generate prediction.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.code(traceback.format_exc(), language="text")
                
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        st.code(traceback.format_exc(), language="text")
        
    finally:
        # Final cleanup
        if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
            try:
                os.unlink(tmp_filename)
            except:
                pass
else:
    # Display example when no model is loaded
    st.info("Please upload a model to begin classification. The model should be trained to output a single value between 0 and 1, where:")
    st.write("• Values closer to 0 indicate human-generated images")
    st.write("• Values closer to 1 indicate AI-generated images")

    # Explain expected model format
    st.subheader("Expected Model Format")
    st.write("Your model should accept RGB image inputs and output a single prediction value. Most binary classification models trained with Keras should work out of the box.")
    
    # Custom objects explanation
    st.subheader("Custom Objects")
    st.write("""
    If your model uses custom layers, losses, or metrics, expand the 'Define Custom Objects' section above and enter their definitions.
    This is necessary for models that use non-standard TensorFlow components.
    """)
    
    # Basic instructions
    st.subheader("How to Use")
    st.write("1. Optionally define any custom objects your model requires")
    st.write("2. Upload your .keras or .h5 model file")
    st.write("3. Wait for the model to load")
    st.write("4. Upload an image to classify")
    st.write("5. View the classification results")
