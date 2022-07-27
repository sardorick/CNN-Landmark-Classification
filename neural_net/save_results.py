from flask import request
import csv
import os
import pandas as pd
from requests import head

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')

def save_results(text, csv_filename='preds.csv', path=UPLOAD_PATH):
    upload_file_request = request.files['image_name']
    filename = upload_file_request.filename
    img_name = f'{filename}.jpg'
    with open(csv_filename, mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([img_name, text])
        
