from flask import Flask, request, render_template
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model
from PIL import Image
import pyrebase

# for uploading and downloading of image to cloud database
config = {
  "apiKey": "AIzaSyCFtWj8lMsvAFcSw2c8uCNRLo44zQhuY4k",
  "authDomain": "vege-image-project.firebaseapp.com",
  "projectId": "vege-image-project",
  "storageBucket": "vege-image-project.appspot.com",
  "messagingSenderId": "476931126209",
  "appId": "1:476931126209:web:b710a85a2747c5fde37540",
  "measurementId": "G-SQTBN3Z860",
  "databaseURL":"https://console.firebase.google.com/project/vege-image-project/storage/vege-image-project.appspot.com/files"
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
path_on_cloud = "vege images/"




# laod model
model = load_model('vege_model.h5')

# labels for image prediction
labels = {0: 'Choy Sum', 1: 'Coriander', 2: 'Cucumber', 3: 'Garlic', 4: 'Green Chili', 5: 'Green bean / String bean',
          6: 'Holland Sweet Potato', 7: 'Long Beans', 8: 'Long Eggplant', 9: 'Luffa', 10: 'Okra',
          11: 'Red Chili Kulai / Kulai Hybrid', 12: 'Red Onion', 13: 'Round Cabbage', 14: 'Scallions', 15: 'Tomato',
          16: 'Yellow Holland Onion'}

#load price data
vege_price = pd.read_csv("https://raw.githubusercontent.com/bq0722/VgPrice_Pred_App/main/Price%20dataset/vg_price.csv")
# function to get price range
def get_price_range(vege):
    min_price = round(vege_price.avg_min[vege_price.vege_name == vege].iloc[0],2)
    max_price= round(vege_price.avg_max[vege_price.vege_name == vege].iloc[0],2)
    vg_unit= vege_price.unit[vege_price.vege_name == vege].iloc[0]
    price_range = "RM " + str(min_price) + " to RM " + str(max_price) + " per " + vg_unit
    return price_range

# resize the image and predict the image
def processed_img(img_name):
    img=load_img(img_name,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    return res



# Declare a Flask app
app = Flask(__name__)

@app.route('/')
def main():
    return render_template("website.html")

@app.route('/submit',methods=['POST'])
def submit():
    img = request.files['image_upload']
    img_name = img.filename
    img.save(img_name)
    #predict image
    p = processed_img(img_name)
    #predict price
    price = get_price_range(p.lower())

    # upload to cloud database
    img_path = path_on_cloud + img_name
    storage.child(img_path).put(img_name)
    #get url of image on cloud for displaying the image on website
    img_url = storage.child(img_path).get_url(None)

    return render_template("website.html", prediction=p, price = price, img_url=img_url)



# Running the app
if __name__ == '__main__':
    app.run(debug = True)


