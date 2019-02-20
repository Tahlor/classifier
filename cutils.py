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
    """ Check if check point is a directory. If it is, create a new checkpoint in the directory that increments previous.
        Return unchanged if anything else

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

def get_max_file(path, ignore=None):
    """ Gets the file with the highest (first) number in the string, ignoring the "ignore" string
    Args:
        path (str): Folder to search
    Returns:

    """
    if ignore:
        filtered = [p for p in os.listdir(path) if not re.search(ignore, p)]
    else:
        filtered = os.listdir(path)
    numbers = [(int(re.search("^[0-9]+", p)[0]), p) for p in filtered if re.search("^[0-9]+", p)]
    n, npath = max(numbers) if numbers else (0, "")
    #print("Last File Version: {}".format(npath))
    return n, os.path.join(path, npath)


def increment_path(name = "", base_path="./logs", make_directory=False, ignore="partial"):
    # Check for existence
    mkdir(base_path)
    n, npath = get_max_file(base_path, ignore=ignore)

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