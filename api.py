from distutils.log import debug
from flask import Flask
from flask_restful import Api, Resource, reqparse
import db

app1 = Flask(__name__)
api = Api(app1)

class Preds(Resource):
    def get(self):
        return db.get_preds()

api.add_resource(Preds, "/preds")

if __name__ == "__main__":
    app1.run('0.0.0.0')