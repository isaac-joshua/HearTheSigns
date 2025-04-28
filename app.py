import streamlit as st
import cv2
import numpy as np
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from gtts import gTTS
import base64
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_safely(model_path):
    """
    Safely load the model with error handling and batch_shape compatibility fix
    """
    try:
        # First try: Basic load
        model = load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
        return model
    except ValueError as e:
        if 'batch_shape' in str(e):
            logger.warning("Attempting to load model with input_shape instead of batch_shape...")
            try:
                # Custom objects to handle batch_shape to input_shape conversion
                custom_objects = {
                    'CustomModel': tf.keras.Model,
                    'InputLayer': lambda config: tf.keras.layers.InputLayer(
                        input_shape=config['batch_shape'][1:] if 'batch_shape' in config else None,
                        dtype=config.get('dtype', None),
                        sparse=config.get('sparse', False),
                        name=config.get('name', None)
                    )
                }
                
                model = load_model(model_path, compile=False, custom_objects=custom_objects)
                logger.info("Model loaded successfully with input_shape conversion")
                return model
            except Exception as inner_e:
                logger.error(f"Failed to load model with input_shape conversion: {inner_e}")
                st.error("Error loading the model. Please check if the model file is correct.")
                raise
        else:
            logger.error(f"Error loading model: {e}")
            st.error("Error loading the model. Please check if the model file is correct.")
            raise
    except Exception as e:
        logger.error(f"Unexpected error loading model: {e}")
        st.error("Unexpected error loading the model. Please check if the model file is correct.")
        raise

def speak(text):
    """
    Use gTTS for text-to-speech with improved audio handling
    """
    try:
        # Create a unique filename using timestamp with milliseconds
        timestamp = int(time.time() * 1000)  # millisecond precision
        temp_audio_path = f"temp_audio_{timestamp}.mp3"
        
        # Generate audio
        tts = gTTS(text)
        tts.save(temp_audio_path)

        # Read audio file and encode it in Base64
        with open(temp_audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode()

        # Create a unique key for the audio element
        audio_key = f"audio_{timestamp}"
        
        # Embed HTML for auto-playing audio with unique key
        audio_html = f"""
            <audio id="{audio_key}" autoplay="true">
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            <script>
                var audio = document.getElementById("{audio_key}");
                audio.play().catch(function(error) {{
                    console.log("Audio playback failed:", error);
                }});
                
                // Remove the audio element after playing
                audio.onended = function() {{
                    audio.remove();
                }};
            </script>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
        
        # Clean up the temporary file
        try:
            os.remove(temp_audio_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary audio file: {e}")

        # Add a small delay to ensure audio processing
        time.sleep(0.1)
        
    except Exception as e:
        logger.error(f"Error in text-to-speech: {e}")
        st.warning("Unable to play audio feedback")

# Class labels
classes = [
    "Road narrows on right","50 mph speed limit","Attention Please",
    "Beware of children","CYCLE ROUTE AHEAD WARNING", "Dangerous Left Curve Ahead", "Dangerous Right Curve Ahead",
    "End of all speed and passing limits", "Give Way","Go Straight or Turn Right","Go straight or turn left",
    "Keep-Left","Keep-Right","Left Zig Zag Traffic","No Entry","No_Over_Taking",
    "Overtaking by trucks is prohibited", "Pedestrian Crossing","Round-About","Slippery Road Ahead",
    "Speed Limit 20","Speed Limit 30 KMPh","Stop Sign", "Straight Ahead Only","Traffic_signal",
    "Truck traffic is prohibited", "Turn left ahead","Turn right ahead","Uneven Road"
]

@st.cache_resource(show_spinner="Loading model...")
def load_cached_model():
    """Cache the model loading to improve performance"""
    try:
        model_path = "CUSTOMCNNMODELJOSH.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return load_model_safely(model_path)
    except Exception as e:
        logger.error(f"Error in cached model loading: {e}")
        st.error("Failed to load the model. Please check if the model file exists and is correct.")
        raise

def preprocess_frame(frame, target_size=(64, 64)):
    """
    Resize and normalize the frame for prediction.
    """
    try:
        if frame is None:
            raise ValueError("Empty frame received")
        frame = cv2.resize(frame, target_size)
        frame = frame / 255.0  # Normalize pixel values
        return np.expand_dims(frame, axis=0)  # Add batch dimension
    except Exception as e:
        logger.error(f"Error preprocessing frame: {e}")
        st.error("Error preprocessing the image")
        raise

def process_video(video_bytes):
    """Process video file for road sign detection with real-time display"""
    try:
        # Create a temporary file for the video
        temp_video_path = f"temp_video_{int(time.time())}.mp4"
        
        # Write video bytes to temporary file
        with open(temp_video_path, 'wb') as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            st.error("Error opening video file")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        model = load_cached_model()
        
        # Create placeholders for video display and progress
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        last_predicted_class = None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        processed_frames = []



        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = preprocess_frame(frame)
            predictions = model.predict(input_frame, verbose=0)
            predicted_class = classes[np.argmax(predictions)]
            confidence = float(np.max(predictions))

            print("IM herer")
            if predicted_class != last_predicted_class:
                speak(predicted_class)
                last_predicted_class = predicted_class


            # Draw the label with confidence
            label = f"{predicted_class} ({confidence:.2%})"
            cv2.putText(frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw bounding box around the sign
            height, width = frame.shape[:2]
            box_size = min(width, height) // 4
            center_x = width // 2
            center_y = height // 2
            cv2.rectangle(frame, 
                         (center_x - box_size//2, center_y - box_size//2),
                         (center_x + box_size//2, center_y + box_size//2),
                         (0, 255, 0), 2)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frames.append(frame_rgb)
            
            # Display the current frame
            video_placeholder.image(frame_rgb, caption=label, use_container_width=True)
            
            # Audio feedback

            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count} of {total_frames}")

            # Add a small delay to make the video viewable
            time.sleep(1/fps)

        cap.release()

        # Clean up temporary files
        try:
            os.remove(temp_video_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary video files: {e}")
        
        status_text.text("Processing complete!")
        
        # Create animation of processed frames
        st.write("Processed Video:")
        st.image(processed_frames, caption=["Processed Frame"] * len(processed_frames))
        
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        st.error("Error processing the video file")
        raise

def process_image_bytes(image_bytes):
    """Process image from bytes"""
    try:
        # Decode image from bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Error decoding image")
            return

        model = load_cached_model()
        input_image = preprocess_frame(image)
        predictions = model.predict(input_image, verbose=0)
        predicted_class = classes[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Draw the label with confidence
        label = f"{predicted_class} ({confidence:.2%})"
        cv2.putText(image, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw bounding box
        height, width = image.shape[:2]
        box_size = min(width, height) // 4
        center_x = width // 2
        center_y = height // 2
        cv2.rectangle(image, 
                     (center_x - box_size//2, center_y - box_size//2),
                     (center_x + box_size//2, center_y + box_size//2),
                     (0, 255, 0), 2)

        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=label)
        speak(predicted_class)

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        st.error("Error processing the image")
        raise

def process_webcam():
    """Handle webcam input for road sign detection"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam")
            return

        model = load_cached_model()
        stop_button = st.button("Stop Webcam")
        frame_placeholder = st.empty()
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            input_frame = preprocess_frame(frame)
            predictions = model.predict(input_frame, verbose=0)
            predicted_class = classes[np.argmax(predictions)]
            confidence = float(np.max(predictions))
            
            # Draw the label with confidence
            label = f"{predicted_class} ({confidence:.2%})"
            cv2.putText(frame, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw bounding box
            height, width = frame.shape[:2]
            box_size = min(width, height) // 4
            center_x = width // 2
            center_y = height // 2
            cv2.rectangle(frame, 
                         (center_x - box_size//2, center_y - box_size//2),
                         (center_x + box_size//2, center_y + box_size//2),
                         (0, 255, 0), 2)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, caption=label)
            speak(predicted_class)

    except Exception as e:
        logger.error(f"Error in webcam processing: {e}")
        st.error("Error processing webcam feed")
        raise
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

def main():
    st.set_page_config(
        page_title="Road Sign Detection",
        page_icon="🚸",
        layout="wide"
    )
    
    st.title("🚸 Road Sign Detection with Voice Feedback")
    st.write("""
    This application detects road signs in images and videos and provides voice feedback.
    It can process:
    - Images (jpg, png, jpeg)
    - Videos (mp4, avi, mov)
    """)

    # Show TensorFlow version for debugging
    tf_version = tf.__version__
    st.sidebar.info(f"TensorFlow version: {tf_version}")

    input_source = st.radio(
        "Select input source:",
        ["Upload Image", "Upload Video"]
    )

    if input_source == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            process_image_bytes(image_bytes)

    elif input_source == "Upload Video":
        uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            st.write("Processing video... Please wait.")
            video_bytes = uploaded_video.read()
            process_video(video_bytes)

    elif input_source == "Use Webcam":
        process_webcam()

    # Add a footer with additional information
    st.markdown("---")
    st.markdown("""
    **Note:** The system will automatically:
    - Detect road signs in real-time
    - Provide voice feedback for detected signs
    - Show confidence scores for predictions
    - Display bounding boxes around detected signs
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error("An unexpected error occurred. Please try again.")