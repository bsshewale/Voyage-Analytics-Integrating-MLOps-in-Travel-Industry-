import logging
import os
from datetime import datetime


def get_logger(name="flight_price_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir,
            f"{datetime.now().strftime('%Y-%m-%d')}.log"
        )

        file_handler = logging.FileHandler(log_file)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )

        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance (used everywhere)
logger = get_logger()
