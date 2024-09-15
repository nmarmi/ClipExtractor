"""Contain monitoring information: logs, errors, etc"""

import logging
import sys
from datetime import datetime
from pathlib import Path

def config_logger(
    log_dir: Path = None, level: int = logging.DEBUG
) -> logging.Logger:
    """Logging function. It has two main handlers:

        - a StreamHandler that logs INFO level log in stdout
        - an optional FileHandler that logs DEBUG level log in a .log file

    Args:
        log_dir (Path): output directory where to store log.
            If not passed, logs will only be printed in stdout
        level (int): logging level (default logging.DEBUG)

    Returns:
        logger: main logger
    """
    # Set to Warning all loggers imported from libraries
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.WARNING)

    logger = logging.getLogger()
    # clear any StreamHandler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    # adding handler to stdout
    logger.setLevel(level)  # DEBUG as default
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s - [%(module)s] %(message)s"
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    logger.addHandler(stdout_handler)

    # adding handler to .log file
    if log_dir is not None:
        # add timestamp to log path
        timestamp = datetime.today().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"logs_{timestamp}.log"

        # set .log path
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)

        # log contains also debug
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger