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


def checkpoint(path):
    if os.path.isdir(path) or not os.path.exists(path):
        mkdir(path)
        return os.path.join(path, "1.pt")
    else:
        return path