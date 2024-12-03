import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

st.markdown("# Skin Disease Classification Model Using Convolutional Neural Network")
st.image("skin.jpeg")
paragraph = ""
with open('file.txt', 'r') as f:
    paragraph = f.read()
st.write(paragraph)
    
file = st.file_uploader("Upload File")

if file is not None:
    # To read file as string:
    # To read file as bytes:
    img = Image.open(file)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    img = img.resize((210, 360))
    # x = keras.preprocessing.image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)

    # # tensor = np.vstack([x])
    # tensor = x.astype(np.float32)

    x = np.array(img)  # Convert to NumPy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x.astype(np.float32) 

    print(x.shape)

    interpreter = tf.lite.Interpreter(model_path='model_full (1).tflite')
    classify_lite = interpreter.get_signature_runner('serving_default')
 
    predictions_lite = classify_lite(keras_tensor_4=x)['output_0']
    score_lite = tf.nn.softmax(predictions_lite)

    class_names = ['Acne and Rosacea',
 'Actinic Keratosis, Basal Cell Carcinoma and other Malignant Lesions',
 'Atopic Dermatitis',
 'Bullous Disease',
 'Cellulitis, Impetigo and other Bacterial Infections',
 'Eczema',
 'Exanthems and Drug Eruptions',
 'Hair Loss, Alopecia and other Hair Diseases',
 'Herpes, HPV and other STDs',
 'Light Diseases and Disorders of Pigmentation']
  
    st.markdown("### Results")

    score = "This image most likely belongs to <strong>{}</strong> with a {:.2f} percent confidence.".format(class_names[np.argmax(score_lite)], 100 * np.max(score_lite))
    st.markdown(score,unsafe_allow_html=True)
    
    for i in range(10):
        st.markdown(f"<strong>{class_names[i]}</strong>:  {100 * score_lite[0][i]:.2f}%", unsafe_allow_html=True)

for i in range(5):
    st.markdown("</br>",unsafe_allow_html=True)
st.markdown('<em><strong>Built by Ryheeme Donegan, Rachelle Williams, Kimberly Pecco and Alex Salmon <br>Built with Streamlit, Google Colab, TensorFlow and TensorFlow Lite </br>Dataset used: [Dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) from Kaggle</strong> </em>', unsafe_allow_html=True)