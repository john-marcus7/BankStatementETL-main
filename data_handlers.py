"""Module for handling data from SQLite database and generating reports."""
from BankETL import ETL_BofA_AI
from sqlalchemy import create_engine
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column,Integer, Float, String, DateTime

DB_PATH = "data/transactions.sqlite"
    
Base = declarative_base()

class Transaction(Base):
    """
    A class representing a financial transaction.

    Attributes:
    - id (int): The unique identifier of the transaction.
    - date (datetime): The date of the transaction.
    - description (str): The description of the transaction.
    - amount (float): The amount of the transaction.
    - account_name (str): The name of the account associated with the transaction.
    - transaction_type (str): The type of the transaction (e.g., debit, credit).
    - category_name (str): The category of the transaction.

    Methods:
    - load_transactions(session, start_time, end_time): Loads transactions from the database within a specified time range.
    - update_transactions(session, index, column, value): Updates a specific column of a transaction.
    - save_transactions(session, transactions): Saves transactions to the database.
    - delete_transactions(session, id): Deletes a transaction from the database.
    - get_expenses_by_category(session, start_time, end_time): Retrieves expenses grouped by category within a specified time range.
    - get_income_vs_expenses(session, start_time, end_time): Retrieves the total income and expenses within a specified time range.
    - get_monthly_expenses_trends(session, start_time, end_time): Retrieves monthly expenses trends within a specified time range.
    - get_top_expenses(): Retrieves the top expenses.
    - get_savings_rate(): Retrieves the savings rate.
    """

    engine = create_engine(f'sqlite:///{DB_PATH}')
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime)
    description = Column(String)
    amount = Column(Float)
    account_name = Column(String)
    transaction_type = Column(String)
    category_name = Column(String)

    ### decoretor ###
    def session(func):
        def wrapper(self, *args, **kwargs):
            Session = sessionmaker(bind=self.engine)
            session = Session()
            result = func(self, session, *args, **kwargs)
            session.close()
            return result
        return wrapper
    
    def __repr__(self):
        return f"Transaction(date={self.date}, description={self.description}, amount={self.amount}, account_name={self.account_name}, transaction_type={self.transaction_type}, category_name={self.category_name})"


    ### static methods ###
    
    @staticmethod
    def __convert_to_dataframe(result:list,columns:list)->pd.DataFrame:
        return pd.DataFrame(result, columns=columns)
    
    ### Transactions CRUD ###
    @session
    def load_transactions(self,session,start_time:datetime,end_time:datetime,categories:list,account_name:list) -> None:
        """
        Loads transactions from the database within a specified time range.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.

        Returns:
        - result (pd.DataFrame): The loaded transactions as a pandas DataFrame.
        """
        result =  session.query(Transaction.date,
                                Transaction.description,
                                Transaction.amount,
                                Transaction.account_name,
                                Transaction.category_name)\
            .filter(Transaction.date.between(start_time, end_time))\
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        result = self.__convert_to_dataframe(result,columns=["date","description","amount","account_name","category_name"])
        return result
    
    @session
    def get_categories(self,session)->list:
        """
        Retrieves all categories from the database.

        Parameters:
        - session (Session): The database session.

        Returns:
        - result (list): The categories as a list.
        """
        result = session.query(Transaction.category_name).distinct().all()
        return [item[0] for item in result]
    
    @session
    def get_accounts(self,session)->list:
        """
        Retrieves all accounts from the database.

        Parameters:
        - session (Session): The database session.

        Returns:
        - result (list): The accounts as a list.
        """
        result = session.query(Transaction.account_name).distinct().all()
        return [item[0] for item in result]
    
    @session
    def update_transactions(self,session,index:int,column:str,value)->None:
        """
        Updates a specific column of a transaction.

        Parameters:
        - session (Session): The database session.
        - index (int): The index of the transaction to update.
        - column (str): The name of the column to update.
        - value: The new value for the column.
        """
        session.query(Transaction).filter(Transaction.id == index).update({column: value})
        session.commit()

    @session
    def save_transactions(self,session,transactions:pd.DataFrame)->None:
        """
        Saves transactions to the database.

        Parameters:
        - session (Session): The database session.
        - transactions (pd.DataFrame): The transactions to save as a pandas DataFrame.
        """
        transactions.to_sql("transactions", con=self.engine, if_exists="append", index=False)
        session.commit()

    @session
    def delete_transactions(self,session,id:int)->None:
        """
        Deletes a transaction from the database.

        Parameters:
        - session (Session): The database session.
        - id (int): The ID of the transaction to delete.
        """
        session.query(Transaction).filter(Transaction.id == id).delete()
        session.commit()

    ### Reports ### 
    @session
    def get_expenses_by_category(self,session,start_time:datetime,end_time:datetime,categories:list,account_name:list)->pd.DataFrame:
        """
        Retrieves expenses grouped by category within a specified time range.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.
        - categories (list): The categories to retrieve.

        Returns:
        - result (pd.DataFrame): The expenses grouped by category as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["category_name","amount"])
        result = session.query(Transaction.category_name,Transaction.amount) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.transaction_type == "debit") \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["category_name","amount"])
        result = self.__convert_to_dataframe(result, columns=['category_name','amount'])
        result = result.groupby("category_name").sum().reset_index()
        result['amount'] = abs(result['amount'])
        return result
        
        
    @session
    def get_income_vs_expenses(self,session,start_time:datetime,end_time:datetime, categories:list,account_name:list)->pd.DataFrame:
        """
        Retrieves the total income and expenses within a specified time range.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.
        - categories (list): The categories to retrieve.

        Returns:
        - result (pd.DataFrame): The total income and expenses as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["date","income","expenses"])
        result = session.query(Transaction.date,Transaction.transaction_type,Transaction.amount) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["date","income","expenses"])
        result = self.__convert_to_dataframe(result,columns=["date","transaction_type","amount"])
        result["date"] = result["date"].dt.strftime("%Y-%m")
        result = result.groupby(["date","transaction_type"]).sum().reset_index()
        result["amount"] = abs(result["amount"])
        result = result.pivot(index="date",columns="transaction_type",values="amount").reset_index()
        result = result.rename(columns={"credit":"income","debit":"expenses"})
        if "income" not in result.columns:
            result["income"] = 0
        if "expenses" not in result.columns:
            result["expenses"] = 0
        result.columns.name = None
        return result 

    @session
    def get_monthly_expenses_trends(self,session,start_time:datetime,end_time:datetime, categories:list, account_name:list)->pd.DataFrame:
        """
        Retrieves monthly expenses trends within a specified time range.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.
        - categories (list): The categories to retrieve.

        Returns:
        - result (pd.DataFrame): The monthly expenses trends as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["date","amount"])
        result = session.query(Transaction.date,Transaction.amount) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.transaction_type == "debit") \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["date","amount"])
        result = self.__convert_to_dataframe(result,columns=["date","amount"])
        result["date"] = result["date"].dt.strftime("%Y-%m")
        result = result.groupby("date").sum().reset_index()
        result["amount"] = abs(result["amount"])
        return result

    @session
    def get_daily_expenses_trends(self,session,start_time:datetime,end_time:datetime, categories:list, account_name:list)->pd.DataFrame:
        """
        Retrieves daily expenses trends within a specified time range.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.
        - categories (list): The categories to retrieve.

        Returns:
        - result (pd.DataFrame): The daily expenses trends as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["date","amount"])
        result = session.query(Transaction.date,Transaction.amount) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.transaction_type == "debit") \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["date","amount"])
        result = self.__convert_to_dataframe(result,columns=["date","amount"])
        result["date"] = result["date"].dt.strftime("%Y-%m-%d")
        result = result.groupby("date").sum().reset_index()
        result["amount"] = abs(result["amount"])
        return result
            

    @session
    def get_top_expenses(self,session,start_time:datetime,end_time:datetime,categories:list,account_name:list,top_n = 10)->pd.DataFrame:
        """
        Retrieves the top expenses.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.
        - categories (list): The categories to retrieve.
        - top_n (int): The number of top expenses to retrieve.

        Returns:
        - result (pd.DataFrame): The top expenses as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["short_description","description","amount"])
        result = session.query(Transaction.description,Transaction.amount) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.transaction_type == "debit") \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["short_description","description","amount"])
        result = self.__convert_to_dataframe(result,columns=["description","amount"])
        result.drop_duplicates(subset=["description"],inplace=True)
        result = result.groupby("description").sum().reset_index()
        result["short_description"] = result["description"].str[:15]  # Get only the first 15 characters
        result["short_description"] += result.index.astype(str) + "..." # Add ellipsis and unique index
        result["amount"] = abs(result["amount"])
        result = result.sort_values(by="amount",ascending=False).head(top_n)
        result = result.sort_values(by="amount",ascending=True)
        return result

    
    @session
    def get_savings_rate_per_month(self,session,start_time:datetime,end_time:datetime,categories:list,account_name:list)->pd.DataFrame:
        """
        Retrieves the savings rate per month.

        Parameters:
        - session (Session): The database session.
        - start_time (datetime): The start time of the time range.
        - end_time (datetime): The end time of the time range.

        Returns:
        - result (pd.DataFrame): The savings rate per month as a pandas DataFrame.
        """
        if len(categories) == 0 or len(account_name) == 0:
            return pd.DataFrame(columns=["date","savings_rate"])
        result = session.query(Transaction.date,Transaction.amount,Transaction.transaction_type) \
            .filter(Transaction.date.between(start_time, end_time)) \
            .filter(Transaction.category_name.in_(categories))\
            .filter(Transaction.account_name.in_(account_name)).all()
        if len(result) == 0:
            return pd.DataFrame(columns=["date","savings_rate"])
        result = self.__convert_to_dataframe(result,columns=["date","amount","transaction_type"])
        result["date"] = result["date"].dt.strftime("%Y-%m")
        result = result.groupby(["date","transaction_type"]).sum().reset_index()
        result["amount"] = abs(result["amount"])
        result = result.pivot(index="date",columns="transaction_type",values="amount").reset_index()
        result = result.rename(columns={"credit":"income","debit":"expenses"})
        result.columns.name = None
        if "income" not in result.columns:
            return pd.DataFrame(columns=["date","savings_rate"])
        if "expenses" not in result.columns:
            return pd.DataFrame(columns=["date","savings_rate"])
        result["savings"] = result["income"] - result["expenses"]
        result["savings_rate"] = result["savings"] / result["income"] * 100
        result["savings_rate"] = result["savings_rate"].round(2)
        result = result[["date","savings_rate"]]
        return result

    def execute_etl(self)->None:
        bofa_etl = ETL_BofA_AI()    
        data_found = bofa_etl.load_data()
        
        if not data_found:    
            return

        bofa_etl.ai_categorization()
        
        bofa_etl.save_transactions_to_db()
        
        bofa_etl.delete_processed_csv_files()

    def execute_etl_in_new_file(self,uploaded_file)->dict:
        result = {"status":None,"message":""}
        try:
            bofa_etl = ETL_BofA_AI()    
            data_found = bofa_etl.load_new_item(uploaded_file)
            
            if not data_found:    
                result["status"] = "warning"
                result["message"] = "No new data found in the uploaded file."
                return result

            bofa_etl.ai_categorization()
            
            data_saved = bofa_etl.save_transactions_to_db()
            print(f"data_saved: {data_saved}")
            if not data_saved:
                result["status"] = "warning"
                result["message"] = "No new data to be saved in the database! items already included there."
                return result

            result["status"] = "success"
            result["message"] = "Data successfully added to the database!"
        except Exception as e:
            result["status"] = "error"
            result["message"] = f"{str(e)}"
        return result