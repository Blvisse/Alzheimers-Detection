import tensorflow as tf
import streamlit as st 
import cv2
from PIL import Image,ImageOps
import numpy as np

model = ''

st.write(" # Alzheimers Disease classification system ")

st.write(" Classify the image ")

scan=st.file_uploader("Upload the image file",type=["jpg","png"])

def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if scan is None:
    st.text("Please upload an image file")
else:
    image = Image.open(scan)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    

