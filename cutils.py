import yaml
import os
import fnmatch
import re

def get_config(config="./config/main.yaml"):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass


def checkpoint(path, name=None):
    """
    Path - a folder - create next new checkpoint
    Path - do nothing
    """
    if os.path.isdir(path) or not os.path.exists(path):
        mkdir(path)
        return increment_path(name, path, make_directory=False)
    else:
        return path

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    matching_files_and_folders = fnmatch.filter(os.listdir(base), pattern)
    return len(matching_files_and_folders)>0

def increment_path(name = "", base_path="./logs", make_directory=False):
    # Check for existence
    mkdir(base_path)
    numbers = [int(re.search("^[0-9]+", path)[0]) for path in  os.listdir(base_path) if re.search("^[0-9]+", path)]
    print(numbers)
    n = max(numbers)+1 if numbers else 1

    # Create
    logdir = os.path.join(base_path, "{:02d}_{}".format(n,name))
    if make_directory:
        mkdir(logdir)
    return logdir


if __name__=='__main__':
    x = increment_path("vgg16.pt", base_path="logs")
    print(x)