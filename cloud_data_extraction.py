import yaml
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
yaml_credentials = r"C:\Users\Dr Dankali\AICORE\EDA_France\credentials.yaml"
def cred_reader(credentials):
    with open(credentials, "r") as f:
        creds = yaml.load(f, Loader = yaml.FullLoader)
        return creds

class RDSDatabaseConnector():
    def __init__(self, creds):
        self.creds = creds
    
    def alchemy_egnine(self):
        self.DATABASE_TYPE = 'postgresql'
        self.DBAPI = 'psycopg2'
        self.HOST = self.creds["RDS_HOST"]
        self.USER = self.creds["RDS_USER"]
        self.PASSWORD = self.creds["RDS_PASSWORD"]
        self.DATABASE = self.creds["RDS_DATABASE"]
        self.PORT = self.creds["RDS_PORT"]
        self.engine = create_engine(f"{self.DATABASE_TYPE}+{self.DBAPI}://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}")
        self.engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        inspector = inspect(self.engine)
        self.tables = inspector.get_table_names()
        print(self.tables)
        
        

    def df_from_RDS(self):
        with self.engine.connect() as connection:
                self.loan_payments = pd.read_sql_table("loan_payments", connection)
                return self.loan_payments.head(10)
        
    def df_to_csv(self):
        self.file_path = r"C:\Users\Dr Dankali\AICORE\EDA_France\loan_payments.csv"
        self.loan_payments.to_csv(self.file_path, index=False)

creds = cred_reader(yaml_credentials)
payments1 = RDSDatabaseConnector(creds)
payments1.alchemy_egnine()
payments1.df_from_RDS()
payments1.df_to_csv()

# from sqlalchemy import inspect
# inspector = inspect(engine)
# inspector.get_table_names()