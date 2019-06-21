from model._globals import Config
import logging
import os
import sys

config = Config('config.yaml')


def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

    paths = ['result', 'result/%s' % _id, 'result/%s/models' % _id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)


def setup_logging(config, _id):
    '''Configure logging object to track parameter settings,training, and evaluation.

    Args:
        config(object):Global object specifying system runtime params.

    Returns:
        logger(object):Logging object
    '''

    logger = logging.getLogger(_id)
    hdlr = logging.FileHandler('result/%s/params.log' % _id)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    logger.info("Runtime params:")
    logger.info("----------------")
    for attr in dir(config):
        if not "__" in attr and not attr in ['header', 'date_format', 'path_to_config', 'build_group_lookup']:
            logger.info('%s: %s' % (attr, getattr(config, attr)))
    logger.info("----------------\n")

    return logger

