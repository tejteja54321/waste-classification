from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model


app = Flask(__name__)


upload_folder = 'static/uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model (replace 'your_model.h5' with the actual model path)
model = load_model('Model2.h5')

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

@app.route('/charts')
def charts():
    return render_template('charts.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return render_template('prediction.html', message="No file part")
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('prediction.html', message="No selected file")
    
    if file and allowed_file(file.filename):
        # Secure the filename and save the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Open the image and preprocess it
        img = Image.open(filepath)
        img = img.resize((224, 224))  # Resize the image to match your model's expected input
        img_array = np.array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]  # Get the predicted class index

        # Assuming you have a class label mapping
        class_labels = ['Cardboard', 'Glass', 'Litter','Metal','Paper','Plastic']  # Replace with your actual class labels
        predicted_label = class_labels[predicted_class]

        # Return the result page with the prediction
        return render_template('result2.html', filename=filename, prediction=predicted_label)

    return render_template('prediction.html', message="Invalid file format")

if __name__ == '__main__':
    app.run(debug=True, port=5001)
