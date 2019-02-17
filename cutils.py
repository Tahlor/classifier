import yaml
import os

def get_config(config="./config/main.yaml"):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

