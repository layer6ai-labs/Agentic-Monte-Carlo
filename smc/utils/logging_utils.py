

import os
import datetime
import logging
from typing import Tuple


def setup_logging() -> Tuple[logging.Logger, str]:
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # create the directory if it doesn't exist

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"smc_run_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # save log
    file_handler = logging.FileHandler(log_filename, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger, timestamp