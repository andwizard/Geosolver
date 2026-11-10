import os
from langchain_core.tools import tool
from utils.logger import get_logger_with_session
from utils.utils import read_yaml
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

@tool
def get_sub_categories_by_category(main_category: str) -> list[str]:
    """
    Get subcategories list by main category
    
    Args:
        main_category (str): Main category name, e.g. "1 Geometric and Related Operations"
        
    Returns:
        list[str]: List of subcategories, returns error message if main category does not exist
    """
    logger = get_logger_with_session()
    logger.info(f"Reading subcategories: main_category: {main_category}")

    # Read catalog configuration file
    catalogue = read_yaml("../book/Geospatial_Analysis/catalogue.yaml")

    # catalogue is a dictionary, first check if main_category exists as a key
    if main_category not in catalogue:
        return f"Main category does not exist, please check input parameters: main_category: {main_category}"

    # Get all subcategories under the main category
    sub_categories = catalogue[main_category]

    # Return subcategories
    return sub_categories

@tool
def get_book_content(main_category: str, sub_category: str) -> str:
    """
    Get content based on main category and subcategory name
    
    Args:
        main_category: Main category name with index, e.g. "1 Geometric and Related Operations"
        sub_category: Subcategory name with index, e.g. "1.1 Length and area for vector data"
        
    Returns:
        str: Content text, truncated if exceeds 5000 characters
    """
    logger = get_logger_with_session()
    # Build markdown file path
    logger.info(f"Reading book content: main_category: {main_category}, sub_category: {sub_category}")
    markdown_path = f"../book/Geospatial_Analysis/{main_category}/{sub_category}.md"
    
    # Check if file exists
    if not os.path.exists(markdown_path):
        return f"File does not exist, please check input parameters: main_category: {main_category} sub_category: {sub_category}"
    
    # Load markdown file
    loader = UnstructuredMarkdownLoader(markdown_path)
    data = loader.load()
    
    # Get content
    content = data[0].page_content
    
    # Truncate if exceeds 5000 characters
    if len(content) > 5000:
        content = content[:5000]
        
    return content

