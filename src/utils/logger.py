import logging

def get_logger(fileName=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formater = logging.Formatter(
        fmt='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S')

    handlers = [logging.StreamHandler()]
    if fileName is not None:
        handlers.append(logging.FileHandler(fileName, mode='w'))
    for handler in handlers:
        handler.setFormatter(formater)
        logger.addHandler(handler)
    
    return logger