import yaml
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect

# Path to .yaml file with database credentials
yaml_credentials = r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\credentials.yaml"

# Fucntion to extract credential dictionary
def cred_reader(credentials):
    with open(credentials, "r") as f:
        creds = yaml.load(f, Loader = yaml.FullLoader)
        return creds

# This class will connect to the RDS database
class RDSDatabaseConnector():
    def __init__(self, creds):
        self.creds = creds
    # This method will take the credentials dictionary and create an SQLAlchemy engine object
    def alchemy_egnine(self):
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.HOST = self.creds["RDS_HOST"]
        self.USER = self.creds["RDS_USER"]
        self.PASSWORD = self.creds["RDS_PASSWORD"]
        self.DATABASE = self.creds["RDS_DATABASE"]
        self.PORT = self.creds["RDS_PORT"]
        # Here we create and engine object for the PostgreSQL database
        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")
        # We connect to the databse and print all the tables
        self.engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        inspector = inspect(self.engine)
        self.tables = inspector.get_table_names()
        print(self.tables)
        
        
    # This method will make a Pandas dataframe from the RDS database and print the first 10 rows to check its worked correctly
    def df_from_RDS(self):
        with self.engine.connect() as connection:
                self.loan_payments = pd.read_sql_table("loan_payments", connection)
                return self.loan_payments.head(10)

    # This method will write the dataframe to a .csv file for later use    
    def df_to_csv(self):
        self.file_path = r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\loan_payments.csv"
        self.loan_payments.to_csv(self.file_path, index=False)

# Read credentials from .yaml file
creds = cred_reader(yaml_credentials)
# Create an insatnce of RDSDatabaseConnector
payments1 = RDSDatabaseConnector(creds)
# Create an engine and connect to the database
payments1.alchemy_egnine()
# Create a dataframe from the database
payments1.df_from_RDS()
# Save the dataframe to .csv format for later use
payments1.df_to_csv()
