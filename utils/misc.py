import os
import atexit
import logging

def create_logger(log_filename, env_vars):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    output_dir = env_vars.get("output_folder_path", "./outputs")
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, f"{log_filename}.log")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Ensure logs are flushed at exit
    def flush_logs():
        for handler in logger.handlers:
            handler.flush()
    atexit.register(flush_logs)

    return logger
