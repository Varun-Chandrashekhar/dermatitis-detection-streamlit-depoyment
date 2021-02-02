
import fastai


#Streamlit

#Importing Libraries
import PIL
#import os
#import cv2
#import urllib

import streamlit as st # Web application
from PIL import Image, ImageOps # Image processing
import requests # image URL requesting
#from io import BytesIO # Image conversion

import numpy as np # linear algebra
import pandas as pd # data processing

from fastai import *
from fastai.vision import *
from fastai.imports import *
import fastai.vision.all

path = Path('dermatitis')

#Setting up Variables
classes = ['eczema','measles','melanoma']

#Create a title and a sub-title
st.title("Dermatitis Detection")
st.write("Helps detect Skin Diseases using Machine Learning")

#Open and Display an Image
st1_image = PIL.Image.open('Dermatitis Detection.png')
st.image(st1_image)

# Uploading Image File
#Set warnings to False
st.set_option('deprecation.showfileUploaderEncoding',False)

#User Input
st.subheader("User Input:")
#Uploading image file
file = st.file_uploader("Please upload image of skin", type=["jpg","png"])#, use_column_width=True)

if file is None:
  st.text ("Please upload an image file to continue ...")
else:
  image_input = PIL.Image.open(file)
  #download_images(file, 'User Input')
  st.image(image_input)#, use_column_width=True)
  #Save Image
  image_input.save('00000000.jpg', 'JPEG')#open_image('00000000.jpg', 'JPG')
  #Setting Up The Text for Classification
  with st.spinner('Processing Image.....'):
    path = Path('dermatitis')
    data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
            ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
    with st.spinner('Exporting Model.....'):
      learn = cnn_learner(data, models.resnet34, metrics=error_rate)
      learn.load('stage-4')
      learn.export()
      defaults.device = torch.device('cpu')
      learn = load_learner(path)
      with st.spinner('Classifying.....'):
        user_input_img = open_image('00000000.jpg')
        pred_class,pred_idx,outputs = learn.predict(user_input_img)
        st.subheader("Prediction : ")
        st.success(str(pred_class))
        accuracy = outputs[pred_idx]
        accuracy = str(accuracy)
        accuracy = accuracy[9:-1]
        accuracy = int(accuracy)
        accuracy = accuracy/100
        accuracy = str(accuracy)
        st.subheader("Accuracy: ")
        st.success(accuracy+"%")


  #Set a subheader
  st.subheader('Confusion Matrix: ')
  # Show  the data as a image
  st2_image = PIL.Image.open('Confusion_Matrix.PNG')
  st.image(st2_image, caption = 'Confusion Matrix (Showing Accuracy of Machine Learning Model)')#, use_column_width=False)

  #Set a subheader
  st.subheader('Sample Images: ')
  # Show  the data as a image
  st3_image = PIL.Image.open('Data_1.PNG')
  st.image(st3_image, caption = 'Sample Images')#, use_column_width=False)