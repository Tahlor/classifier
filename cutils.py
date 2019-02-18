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

def get_max_file(path):
    """
    Args:
        path:

    Returns:

    """
    numbers = [(int(re.search("^[0-9]+", path)[0]),path) for path in os.listdir(path) if re.search("^[0-9]+", path)]
    n, npath = max(numbers) if numbers else (0, "")
    return n, os.path.join(path, npath)


def increment_path(name = "", base_path="./logs", make_directory=False):
    # Check for existence
    mkdir(base_path)
    n, npath = get_max_file(base_path)

    # Create
    logdir = os.path.join(base_path, "{:02d}_{}".format(n+1,name))
    if make_directory:
        mkdir(logdir)
    return logdir


if __name__=='__main__':
    x = increment_path("vgg16.pt", base_path="logs")
    print(x)
    y = get_max_file("./checkpoints/vgg16")
    print(y)