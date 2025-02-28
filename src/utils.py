from src.exception import CustomException
from src.logger import logging
import sys
import os
import pickle

import numpy as np
import pandas as pd
import dill


def save_object(file_path, obj):
    try:

        logging.info("Saving Started")


        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Saving pickle file completed")

    except Exception as e:
        raise CustomException(e, sys)