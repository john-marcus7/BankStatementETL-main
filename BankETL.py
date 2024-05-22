from datamodels import ConfigFile
import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from loguru import logger
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from streamlit import chat_message
import pandas as pd
import os
import json


CONFIG_PATH = os.environ.get('CONFIG_PATH', 'config/etl_bofa_ai_config.json')


class ETL_BofA_AI:
    def __init__(self):
        logger.info("Initializing ETL_BofA_AI")
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)
        logger.info("Config file loaded, validating with pydantic")
        self.config = ConfigFile(**self.config)
        self.selected_llm = self.config.ETL_configs["selected_llm"]
        if self.selected_llm not in self.config.LLM_configs.keys():
            raise ValueError("Selected LLM not found in config file")
        if self.selected_llm == "Ollama_LLM":
            self.llm =  Ollama(**self.config.LLM_configs[self.selected_llm])
        if self.selected_llm == "Open_AI_LLM":
            self.llm =  ChatOpenAI(**self.config.LLM_configs[self.selected_llm])
        self.prompt_templates_path = self.config.LLM_configs["prompt_templates_path"]  
        
        logger.info("LLM initialized")

        logger.info("loading config parameters")
        self.text_to_replace = self.config.ETL_configs["text_to_replace"]
        self.data_files_path = self.config.ETL_configs["data_files_path"]
        self.db_engine = None
        self.langchain_db_conn = None
        self.__create_or_load_db()
        self.category_df = self.load_categories_from_db()
    
    def __create_or_load_db(self)->None:
        db_path = self.config.ETL_configs["db_path"]
        self.db_engine = create_engine(f'sqlite:///{db_path}')
        self.langchain_db_conn = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        db_connection = sqlite3.connect(db_path)
        cursor = db_connection.cursor()

        scripts_path = self.config.ETL_configs["db_scripts_path"]
        create_tables_script = scripts_path + self.config.ETL_configs["db_scripts"]["create_tables"]

        with open(create_tables_script, 'r') as f:
            sql_script = f.read()

        cursor.executescript(sql_script)

        db_connection.commit()
        db_connection.close()

    def __get_schema(self,_):
        return self.langchain_db_conn.get_table_info()

    def __run_query(self,query):
        return self.langchain_db_conn.run(query.strip("`"))

    def save_new_categories_to_db(self,df:pd.DataFrame)->None:
        if self.db_engine is None:
            raise ValueError("DB engine is not initialized")
        logger.info("Saving categories to db")
        df.to_sql('categories', con=self.db_engine, if_exists='append', index=False)
        logger.info(f"Categories saved to db, {len(df)} rows added") 

    def __update_categories_in_db(self,df:pd.DataFrame)->None:
        if self.db_engine is None:
            raise ValueError("DB engine is not initialized")
        logger.info("Updating categories in db")

        df = df[['category_name','description','account_name']]
        df.columns = ['category_name','pattern','account_name']

        original_categories = pd.read_sql_table('categories', con=self.db_engine)
        original_categories.drop(columns=['id'], inplace=True)

        new_categories = df[~df.isin(original_categories)].dropna()
        new_categories.drop_duplicates(inplace=True)

        new_categories.to_sql('categories', con=self.db_engine, if_exists='append', index=False)
        logger.info(f"Categories updated in db, {len(df)} rows updated")

    def load_categories_from_db(self)->pd.DataFrame:
        if self.db_engine is None:
            raise ValueError("DB engine is not initialized")
        logger.info("Loading categories from db")
        df = pd.read_sql_table('categories', con=self.db_engine)
        logger.info(f"Categories loaded from db, {len(df)} rows loaded")
        return df
    
    def load_transactions_from_db(self)->pd.DataFrame:
        if self.db_engine is None:
            raise ValueError("DB engine is not initialized")
        logger.info("Loading transactions from db")
        df = pd.read_sql_table('transactions', con=self.db_engine)
        logger.info(f"Transactions loaded from db, {len(df)} rows loaded")
        return df
    
    def load_data(self)-> bool:
        logger.info("Loading data")
        csv_files = self.__get_csv_files(self.data_files_path)
        if len(csv_files) == 0:
            logger.info("No csv files found in path")
            return False
        self.df_list = self.__create_dataframe_list(csv_files)
        logger.info(f"Data loaded, {len(self.df_list)} dataframes created")
        
        self.__clean_data_and_apply_first_classification()
        return True
    
    def load_new_item(self,uploaded_file:dict)->bool:
        logger.info("Loading new item")
        if uploaded_file is None:
            logger.info("No file uploaded")
            return False
        self.df_list = self.__create_dataframe_for_new_item(uploaded_file)
        logger.info(f"Data loaded, {len(self.df_list)} dataframes created")
        
        self.__clean_data_and_apply_first_classification()
        return True

    def __create_dataframe_for_new_item(self,uploaded_file)->pd.DataFrame:
        dataframe_list = []
        account_type = None
        for item in self.config.ETL_configs["account_name_pattern"]:
            if item["pattern"] in uploaded_file.name:
                account_type = item
                break
        if account_type is None:
            raise ValueError("Account type not found in config file")
        if "skiprows"in account_type.keys():
            dataframe = pd.read_csv(uploaded_file, skiprows=account_type["skiprows"])
        else:
            dataframe = pd.read_csv(uploaded_file)
            dataframe["account_name"] = account_type["account_name"]
            dataframe_list.append({
                "account_name":account_type["account_name"],
                "dataframe":dataframe,
                "method":account_type["method"],
                "categories":account_type["categories"]
                })
            logger.info(f"Dataframe created for {uploaded_file} with account type {account_type['account_name']}")
        return dataframe_list  

    
    @staticmethod
    def __get_csv_files(path):
        csv_files = []
        for file in os.listdir(path):
            if file.endswith(".csv"):
                csv_files.append(os.path.join(path, file))
        logger.info(f"Found {len(csv_files)} csv files in path {path}")
        return csv_files
    
    def __create_dataframe_list(self,csv_files):
        dataframe_list = []
        for file in csv_files:
            account_type = None
            for item in self.config.ETL_configs["account_name_pattern"]:
                if item["pattern"] in file:
                    account_type = item
                    break
            if account_type is None:
                raise ValueError("Account type not found in config file")
            if "skiprows"in account_type.keys():
                dataframe = pd.read_csv(file, skiprows=account_type["skiprows"])
            else:
                dataframe = pd.read_csv(file)
            dataframe["account_name"] = account_type["account_name"]
            dataframe_list.append({
                "account_name":account_type["account_name"],
                "dataframe":dataframe,
                "method":account_type["method"],
                "categories":account_type["categories"]
                })
            logger.info(f"Dataframe created for {file} with account type {account_type['account_name']}")
        return dataframe_list
        
    def __clean_data_and_apply_first_classification(self):
        for item in self.df_list:
            if item["method"] == "transform_checking_data":
               item["dataframe"] = self.__transform_checking_data(item)
            elif item["method"] == "transform_credit_card_data":
               item["dataframe"] = self.__transform_credit_card_data(item)
            elif item["method"] == "None":
               item["dataframe"]['category_name'] = item["dataframe"]['description'].apply(self.__get_transaction_category_from_db, context=item["account_name"])
               item["dataframe"] = item["dataframe"][['date','description','amount','transaction_type','account_name','category_name']]
               logger.info(f"Dataframe transformed for {item['account_name']}")



    def __get_transaction_category_from_db(self,description,context:str):
        categories = self.category_df[self.category_df["account_name"]==context]
        for _, row in categories.iterrows():
            key = row["pattern"]
            if key in description:
                return row["category_name"]
        return None

    def __transform_checking_data(self,item:dict)->pd.DataFrame:
        df = item["dataframe"]
        
        # Convert the 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Remove commas from the 'Amount' column and convert it to float
        if df['Amount'].dtype == object:
            df['Amount'] = df['Amount'].str.replace(',','').astype(float)
        
        # Drop the 'Running Bal.' column
        df.drop(columns=['Running Bal.'], inplace=True)
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Add a new column 'transaction_type' based on the sign of the 'Amount' column
        df['transaction_type'] = df['Amount'].apply(lambda x: 'debit' if x < 0 else 'credit')

        # Rename the columns
        df.columns = ['date','description','amount','account_name','transaction_type']
        
        # Remove payments to the credit card
        df.drop(df[df['description'].str.contains('Online Banking payment to CRD')].index, inplace=True)

        # Classify the transactions into categories from db
        df['category_name'] = df['description'].apply(self.__get_transaction_category_from_db, context=item["account_name"])
        
        # Reorder the columns
        df = df[['date','description','amount','transaction_type','account_name','category_name']]
        
        # Return the transformed dataframe
        logger.info(f"Dataframe transformed for {item['account_name']}")
        return df

    def __transform_credit_card_data(self,item:dict)->pd.DataFrame:
        if item["method"] != "transform_credit_card_data":
            raise ValueError("Method for this item is not transform_credit_card_data")
        df = item["dataframe"]
        
        # Convert the 'Date' column to datetime format
        df['Posted Date'] = pd.to_datetime(df['Posted Date'])
        
        # Remove commas from the 'Amount' column and convert it to float
        if df['Amount'].dtype == str:
            df['Amount'] = df['Amount'].str.replace(',','').astype(float)
        
        # Drop the 'Running Bal.' column
        df.drop(columns=['Reference Number','Address'], inplace=True)
        
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Add a new column 'transaction_type' based on the sign of the 'Amount' column
        df['transaction_type'] = df['Amount'].apply(lambda x: 'debit' if x < 0 else 'credit')

        # Rename the columns
        df.columns = ['date','description','amount','account_name','transaction_type']
        
        # Remove payments to the credit card
        df.drop(df[df['description'].str.contains('Online payment from CHK')].index, inplace=True)

        # Classify the transactions into categories from db
        df['category_name'] =  df['description'].apply(self.__get_transaction_category_from_db, context=item["account_name"])
        
        # Reorder the columns
        df = df[['date','description','amount','transaction_type','account_name','category_name']]
        
        # Return the transformed dataframe
        logger.info(f"Dataframe transformed for {item['account_name']}")
        return df

    def save_transactions_to_db(self)->bool:
        if self.db_engine is None:
            raise ValueError("DB engine is not initialized")
        
        actions = 0
        scripts_path = self.config.ETL_configs["db_scripts_path"]
        latest_transaction_script = scripts_path + self.config.ETL_configs["db_scripts"]["latest_transaction"]

        with open(latest_transaction_script, 'r') as f:
            sql_script = f.read()
        
        latest_transaction = pd.read_sql_query(sql_script, con=self.db_engine)['max_date'][0]

        logger.info("Saving transactions to db")
        for item in self.df_list:
            df = item["dataframe"]
            #if latest_transaction is not None:
            #    df = df[df["date"]>latest_transaction]
            if len(df) == 0:
                logger.info(f'No new transactions for {item["account_name"]}')
                continue
            df.to_sql('transactions', con=self.db_engine, if_exists='append', index=False)
            logger.info(f'Transactions of type {item["account_name"]} saved to db, {len(df)} rows inserted or updated')
            actions += 1
        if actions == 0:
            logger.info("returning false, no transactions saved to db")
            return False
        else:
            return True


    def ai_categorization(self):
        categorization_prompt_file = self.prompt_templates_path + self.config.LLM_configs["prompt_templates"]["categorization"]

        with open(categorization_prompt_file, 'r') as f:
            categorization_prompt = f.read()
    
        
        for item in self.df_list:
            df = item["dataframe"]
            df["category_name"] =  df['description'].apply(self.__get_transaction_category_from_db, context=item["account_name"])
            df_to_fill = df[df["category_name"].isnull()].copy()
            if len(df_to_fill) == 0:
                logger.info(f'No transactions to categorize for {item["account_name"]}')
                continue
            df_to_fill.drop(columns=['category_name'], inplace=True)
            df_to_fill = self.__categorize_transactions_with_llm(df_to_fill,categorization_prompt,categories=item["categories"])
            self.__update_categories_in_db(df_to_fill)
            df['category_name'].fillna(df_to_fill['category_name'], inplace=True)
            item["dataframe"] = df
            logger.info(f'Transactions of type {item["account_name"]} categorized with AI, {len(df_to_fill)} rows categorized')
    
    def __categorize_transactions_with_llm(self,df:pd.DataFrame,categorization_prompt:str,categories:str)->pd.DataFrame:
        transactions = df['description'].to_json()

        category_prompt_template = PromptTemplate.from_template(categorization_prompt)
        prompt = category_prompt_template.format(transactions=transactions, categories=categories)

        if self.selected_llm == "Open_AI_LLM":
            logger.info("Categorizing transactions with OpenAI LLM")
            response = self.llm.invoke(prompt)
            data_categories = json.loads(response.content)
            data_categories = pd.Series(data_categories,name="category_name")
            data_categories.index = df.index
            result =  df.join(data_categories)
            logger.info("Transactions categorized with OpenAI LLM")
            return result
        
        elif self.selected_llm == "Ollama_LLM":
            logger.info("Categorizing transactions with Ollama LLM")
            response = self.llm(prompt)
            data_categories = json.loads(response)
            result = df.join(pd.Series(data_categories,name="category_name"))
            logger.info("Transactions categorized with Ollama LLM")
            return result
        
        else:
            logger.error("LLM not found. Returning original dataframe")
            return df
        
    def delete_processed_csv_files(self):
        for file in os.listdir(self.data_files_path):
            if file.endswith(".csv"):
                os.remove(os.path.join(self.data_files_path, file))
        logger.info(f"processed csv files deleted from path {self.data_files_path}")

    def answer_question_from_db(self,question:str,placeholder_response:chat_message)->str:
        sql_prompt_file = self.prompt_templates_path + self.config.LLM_configs["prompt_templates"]["generate_sql_query"]
        natural_prompt_file = self.prompt_templates_path + self.config.LLM_configs["prompt_templates"]["generate_natural_response"]

        with open(sql_prompt_file, 'r') as f:
            prompt_template_text = f.read()

        with open(natural_prompt_file, 'r') as f:
            natural_template_text = f.read()

        sql_prompt_template = ChatPromptTemplate.from_template(prompt_template_text)
        natural_prompt_template = ChatPromptTemplate.from_template(natural_template_text)

        sql_response = (
            RunnablePassthrough.assign(schema=self.__get_schema)
            | sql_prompt_template
            | self.llm.bind(stop=["\nSQLResult:"])
            | StrOutputParser()
        )

        sql_agent = (
            RunnablePassthrough.assign(query=sql_response)
            | RunnablePassthrough.assign(
                schema=self.__get_schema,
                response=lambda x: self.__run_query(x["query"]),
            )
            | natural_prompt_template
            | self.llm
        )

        assistant_response = ""

        input_question = {"question": question}
        try:
            for chunk in sql_agent.stream(input_question):
                assistant_response += chunk.content
                placeholder_response.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)
        except OperationalError as e:
            assistant_response = f"""
            Sorry. I could not execute the SQL script to answer this.
            I found this error while trying it: 
            {str(e)}"""
            placeholder_response.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)
        except Exception as e:
            assistant_response = f"""
            Sorry. I got an unexpected error.
            This is what the code is telling me that's wrong: 
            {str(e)}

            Please contact the developer to fix this issue.
            Please be patient because his former boss is a jerk and he's still annoying him."""
            placeholder_response.chat_message("assistant").markdown(assistant_response, unsafe_allow_html=True)
        
        return assistant_response