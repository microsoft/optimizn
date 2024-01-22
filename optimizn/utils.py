import logging


def get_logger(logger_name='optimizn_logger'):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    return logger