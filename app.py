from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import Adafruit_GPIO as GPIO
import Adafruit_GPIO.SPI as SPI
import os
import serial
# create a directory called 'tmp' if it doesn't already exist
if not os.path.exists('tmp'):
    os.makedirs('tmp')

app = Flask(__name__, template_folder='template')
model = load_model('C:/Users/nurna/Desktop/project/meat_classify.h5') # replace with your own model file

#code started of sesnor

# Initialize the serial port
ser = serial.Serial('COM5', 9600)

# Define the Flask app and a route to display the sensor data

@app.route('/sensor')
def index():
    return render_template('sensor.html')

@app.route('/get_values')
def get_values():
    # Read the sensor data from serial port
    sensor_data = ser.readline().strip().decode('utf-8')
    
    # Split the sensor data into two values for MQ-4 and M135 sensors
    mq4_value, m135_value = sensor_data.split(',')
    
    return {'mq4_value': mq4_value, 'm135_value': m135_value}

#Code ended for sensor



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # get the image file from the form
        img_file = request.files['image']
        # save the image to a temporary directory
        img_path = 'tmp/' + img_file.filename
        img_file.save(img_path)
        # load the image and preprocess it for the model
        img = image.load_img(img_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.
        # make the prediction
        pred = model.predict(x)
        # get the class label for the prediction
        classes = ['SPOILED', 'FRESH', 'HALF-FRESH']
        # get the class label for the prediction
        class_label = classes[np.argmax(pred)]
        # render the prediction result on the web page
        return render_template('index.html', prediction=f"Prediction is: {class_label}")

if __name__ == '__main__':
    app.run(debug=True)