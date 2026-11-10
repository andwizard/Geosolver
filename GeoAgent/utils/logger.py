import os
import logging
from pathlib import Path

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""
    
    grey = "\x1b[38;21m"       
    yellow = "\x1b[33;21m"     
    red = "\x1b[31;21m"        
    bold_red = "\x1b[31;1m"    
    green = "\x1b[32;21m"      
    blue = "\x1b[34;21m"       
    magenta = "\x1b[35;21m"    
    cyan = "\x1b[36;21m"       
    orange = "\x1b[38;5;208m"  
    reset = "\x1b[0m"          
    
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        "DEFAULT": grey + format + reset,
        "Init": blue + format + reset,
        "Agent": green + format + reset,
        "Node": magenta + format + reset,
        "Router": cyan + format + reset,
        "Tool": orange + format + reset,
        "Main": {
            logging.INFO: orange + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset,
            logging.DEBUG: cyan + format + reset
        },
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        session_id = os.environ.get("SESSION_ID", "default")
        
        if session_id in record.name:
            log_fmt = self.FORMATS["Main"].get(record.levelno, self.FORMATS["DEFAULT"])
        else:
            log_fmt = self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"])
        
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def setup_logger_with_session_id(session_id: str):
    
    logger = logging.getLogger(session_id)
    
    if logger.hasHandlers():
        return logger

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True) 
    
    log_file = log_dir / f"{session_id}.log"
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    
    return logger

def get_logger_with_session():
    """ 初始化日志器和会话ID """
    session_id = os.environ.get("SESSION_ID", "default")
    return setup_logger_with_session_id(session_id)

if __name__ == "__main__":
    for task_num in range(1, 3):
        os.environ["SESSION_ID"] = f"session_{task_num}"

        logger = get_logger_with_session()

        logger.info(f"Starting Task {task_num}")
        logger.debug(f"Debugging Task {task_num}")
        logger.warning(f"Warning in Task {task_num}")
        logger.error(f"Error in Task {task_num}")
        logger.critical(f"Critical issue in Task {task_num}")