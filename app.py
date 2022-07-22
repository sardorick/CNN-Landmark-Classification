from flask import Flask, render_template, request
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


@app.route('/', methods=['POST', 'GET'])
def hello_world():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)
