from BankETL import ETL_BofA_AI
from dotenv import load_dotenv
from loguru import logger
from pathlib import Path
import os
import sys

# Define the log file path inside the container
log_path = "logs/app.log"

# Ensure the log directory exists
Path("logs").mkdir(parents=True, exist_ok=True)


logger.add(
    log_path,
    rotation="1 day",  # Rotate logs daily
    retention="7 days",  # Keep logs for 7 days
    level="INFO",  # Set the minimum log level to INFO
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module} |{message}",
)

# Load environment variables
load_dotenv()

def execute_etl():
    try:
        with open("banner.txt", "r") as f:
            banner = f.read()
        print(banner)
    except FileNotFoundError:
        pass
    logger.info("Starting ETL process")
    bofa_etl = ETL_BofA_AI()

    logger.info("Loading data")
    bofa_etl.load_data()

    logger.info("using AI to categorize transactions")
    bofa_etl.ai_categorization()

    logger.info("saving transactions to db")
    bofa_etl.save_transactions_to_db()

    logger.info("ETL process finished!")

if __name__ == "__main__":
    execute_etl()
