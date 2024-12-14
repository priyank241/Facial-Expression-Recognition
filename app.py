import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load your pre-trained emotion recognition model
model = load_model("ResNetModel.h5")

st.title("Facial Expression Recognition")

# Upload an image
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

def import_and_predict(image_data, model):
    image = load_img(image_data, target_size=(244, 224))
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array /= 255.0
    emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    index_to_emotion = {v: k for k, v in emotion_labels.items()}

    prediction = model.predict(img_array)
    # Get the emotion label with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    return predicted_emotion

if file is None:
    st.text("Please upload an image file")
else:
    emotion_id = import_and_predict(file, model)
    image = load_img(file, target_size=(244, 224))
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    predicted_emotion = emotion_labels.get(emotion_id, "Unknown")
    st.image(image)
    st.header(f"Predicted Emotion: {emotion_id}")
