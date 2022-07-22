from re import L
from sqlalchemy import create_engine
import pandas as pd

def create_db():
    try:
        engine = create_engine("sqlite:///preds.db")
        conn = engine.connect()
        df = pd.read_csv('./static/upload/preds.csv')
        df.to_sql("preds", conn, if_exists="replace")
        conn.close()
        return 0
    except:
        return 1

def get_preds():
    try:
        engine = create_engine("sqlite:///preds.db")
        conn = engine.connect()
        df = pd.read_sql_query("SELECT * FROM preds", conn)
        conn.close()
        return df.to_dict()
    except:
        return {"404": "No data found"}

create_db()