import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = load_model('best_model.keras')  

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    img_file = request.files['image']

    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, 'uploads')

    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, secure_filename(img_file.filename))
    img_file.save(file_path)

    img = image.load_img(file_path, target_size=(224, 224))  
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      

    prediction = model.predict(img_tensor)
    predicted_class = np.argmax(prediction, axis=1)

    labels = ['Anthracnose', 'Downy_Mildew', 'Healthy', 'Mosaic_Virus']  
    result = labels[predicted_class[0]]

    return result

if __name__ == "__main__":
    app.run(debug=True)
