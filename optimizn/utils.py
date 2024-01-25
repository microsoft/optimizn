import logging
import sys


def get_logger(logger_name='optimizn_logger'):
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # log to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s|%(levelname)s|%(message)s'))
    logger.addHandler(handler)
    return logger