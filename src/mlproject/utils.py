import os
import sys
from src.mlproject.expception import CustomException
from src.mlproject.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import pymysql
load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection Established %s", mydb)
        df=pd.read_sql("SELECT * FROM students", mydb)
        print(df.head())
        return df
    except Exception as ex:
        raise CustomException(ex, sys)
