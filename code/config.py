import sys
import os


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))#print(path)
print(project_path)
sys.path.append(project_path)

import logging
from logging import handlers


fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'


def get_logger(log_filename):
    logger = logging.getLogger(log_filename)
    format_str = logging.Formatter(fmt)
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    ch = handlers.RotatingFileHandler(filename=log_filename, mode='a', backupCount=10, maxBytes= 100*1024*1024, encoding='utf-8')
    ch.setFormatter(format_str)
    logger.addHandler(sh)
    logger.addHandler(ch)

    return logger

logger = get_logger("alops.log")


model_path = os.path.join(project_path, "user_data/model_data")
