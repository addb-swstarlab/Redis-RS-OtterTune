from main import main
import logging

class ObjectView(object):
    def __init__(self, dic):
        self.__dict__ = dic

def grid_main(config_dict,logger,log_dir):
    try:
        config = ObjectView(config_dict)
        results = main(config,logger,log_dir)
    except:
        logger.exception("ERROR")
    finally:
        logger.handlers.clear()
        logging.shutdown()
    return results