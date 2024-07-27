import os
import json
import openai
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load_model('best_model.keras')  

# Set your OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Ensure to replace it with a secure approach

# Define labels
labels = ['Anthracnose', 'Downy Mildew', 'Healthy', 'Mosaic Virus']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    img_file = request.files['image']
    if not img_file:
        return jsonify({'error': 'No image provided'}), 400

    basepath = os.path.dirname(__file__)
    upload_folder = os.path.join(basepath, 'uploads', secure_filename(img_file.filename))

    if not os.path.exists(os.path.dirname(upload_folder)):
        os.makedirs(os.path.dirname(upload_folder))

    img_file.save(upload_folder)

    prediction = predict_image(upload_folder)
    response = generate_disease_response(labels[prediction])

    return jsonify(response)

def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.

    prediction = model.predict(img_tensor)
    return np.argmax(prediction, axis=1)[0]

def generate_disease_response(disease):
    if disease == 'Healthy':
        return {"Issue": "No disease detected.", "Explanation of the issue": "The image provided does not contain any plant diseases.", "What to do": "No further action needed."}

    prompt = f"""
    Generate a JSON response with the following format for the disease {disease}:

    {{
        "Issue": "Describe the issue caused by the {disease}.",
        "Explanation of the issue": "Provide a detailed explanation of the {disease}.",
        "What to do": "List actionable steps to manage or treat the {disease}."
    }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant providing detailed information on plant diseases."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        n=1,
        temperature=0.5
    )

    response_json = json.loads(response.choices[0].message['content'].strip())
    return response_json

if __name__ == "__main__":
    app.run(debug=True)
