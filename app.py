from flask import Flask, render_template, request
from flask_restful import Api, Resource
import os
from neural_net.prediction import make_prediction
import torch
from neural_net.model_setup import model
from PIL import Image
import torch
from neural_net.save_results import save_results


app = Flask(__name__)
api = Api(app)
BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
model.load_state_dict(torch.load('model.pth'))


classes = torch.load('classes.pth')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        upload_file_request = request.files['image_name']
        upload_file = Image.open(upload_file_request)
        filename = upload_file_request.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        prediction = make_prediction(model, upload_file, classes)
        save_results(prediction)

        return render_template('index.html', upload=True, upload_image=filename, text=prediction)

    return render_template('index.html', upload=False)



if __name__ == "__main__":
    app.run(debug=True)
