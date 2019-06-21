import yaml
import sys
import os

sys.path.append('../model')


class Config:
    '''Loads parameters from config.yaml into global object'''

    def __init__(self, path_to_config):

        if os.path.isfile(path_to_config):
            pass
        else:
            path_to_config = f"../{path_to_config}"

        setattr(self, "path_to_config", path_to_config)

        dictionary = None

        with open(path_to_config, 'r') as f:
            dictionary = yaml.load(f.read())

        for k, v in dictionary.items():
            setattr(self, k, v)

