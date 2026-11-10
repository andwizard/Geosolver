import os
from langchain_core.tools import tool
import pandas as pd
from typing import Dict, Optional, Annotated, List
from utils.logger import get_logger_with_session

@tool
def reading_csv_summary(data_path: Annotated[str, "Path to the CSV file"] = './data.csv'):
    """
    Read data from a CSV file and return a summary of the table, trying multiple encodings if necessary.

    Args:
        data_path (str): The path to the CSV file.
        
    Returns:
        dict: A dictionary containing the summary of the CSV data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file is empty or too large
        RuntimeError: If file cannot be read with any encoding
    """
    logger = get_logger_with_session()
    
    working_directory = os.environ.get("WORKING_DIRECTORY", os.getcwd())
    if working_directory not in data_path:
        data_path = os.path.abspath(os.path.join(working_directory, data_path))
    
    if not os.path.exists(data_path):
        logger.error(f"(reading_csv_summary) File not found: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")
        
    file_size = os.path.getsize(data_path)
    if file_size == 0:
        logger.error("File is empty")
        raise ValueError("File is empty")
    if file_size > 1_000_000_000:  # 1GB
        logger.error("File is too large (>1GB)")
        raise ValueError("File is too large (>1GB)")
    
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252', 'gbk', 'big5']
    errors = {}
    
    logger.info(f"(reading_csv_summary) Attempting to read CSV file: {data_path}")
    
    for encoding in encodings:
        try:
            data = pd.read_csv(data_path, encoding=encoding)
            if len(data.columns) > 0: 
                logger.info(f"(reading_csv_summary) Successfully read CSV file with encoding: {encoding}")
                
                summary = {
                    'encoding': encoding,
                    'num_rows': len(data),
                    'num_columns': len(data.columns),
                    'column_names': list(data.columns),
                    'column_types': data.dtypes.apply(lambda x: str()).to_dict(),
                    'sample_data': data.head(3).to_dict(orient='records')
                }
                return summary
            else:
                raise ValueError("No columns found in the CSV file")
        except Exception as e:
            errors[encoding] = str(e)
            logger.debug(f"(reading_csv_summary) Failed with encoding {encoding}: {e}")
            continue
    
    error_msg = "Unable to read file with any encoding. Errors:\n" + \
                "\n".join([f"{enc}: {err}" for enc, err in errors.items()])
    logger.error(error_msg)
    raise RuntimeError(error_msg)

@tool
def reading_excel_summary(data_path: Annotated[str, "Path to the Excel file"] = './data.xlsx'):
    """
    Read data from an Excel file and return a summary of the table, trying multiple formats if necessary.
    
    Args:
        data_path (str): The path to the Excel file.
        
    Returns:
        dict: A dictionary containing the summary of the Excel data.
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If file is empty or too large
        RuntimeError: If file cannot be read
    """
    logger = get_logger_with_session()
    
    working_directory = os.environ.get("WORKING_DIRECTORY", os.getcwd())
    if working_directory not in data_path:
        data_path = os.path.abspath(os.path.join(working_directory, data_path))
    
    if not os.path.exists(data_path):
        logger.error(f"(reading_excel_summary) File not found: {data_path}")
        raise FileNotFoundError(f"File not found: {data_path}")
        
    file_size = os.path.getsize(data_path)
    if file_size == 0:
        logger.error("File is empty")
        raise ValueError("File is empty")
    if file_size > 1_000_000_000:  # 1GB
        logger.error("File is too large (>1GB)")
        raise ValueError("File is too large (>1GB)")

    try:
        logger.info(f"(reading_excel_summary) Attempting to read Excel file: {data_path}")
        
        data = pd.read_excel(data_path, sheet_name=None) 
        if not data:
            raise ValueError("Excel file is empty or no valid data found.")
        
        sheet_name = list(data.keys())[0]
        df = data[sheet_name]
        
        summary = {
            'sheet_name': sheet_name,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'column_types': df.dtypes.apply(lambda x: str()).to_dict(),
            'sample_data': df.head(3).to_dict(orient='records')
        }
        
        logger.info(f"(reading_excel_summary) Successfully read Excel file with sheet: {sheet_name}")
        return summary
    
    except Exception as e:
        logger.error(f"(reading_excel_summary) Failed to read Excel file: {e}")
        raise RuntimeError(f"Unable to read Excel file: {e}")