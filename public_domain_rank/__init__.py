import configparser
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_config_filename = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
config = configparser.ConfigParser()
config.read(_config_filename)

from .dataset import dataset
