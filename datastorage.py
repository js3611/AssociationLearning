__author__ = 'joschlemper'

import cPickle as Store
import os.path

'''
Storage Helper

offers two interface: store and retrieve

'''


# Path
root_dir = os.getcwd()
DATA_DIR = "data"
data_dir = "/".join([root_dir, DATA_DIR])

# Creates data directories if they do not exist
if not os.path.isdir(data_dir):
    os.mkdir(DATA_DIR)

# for learning_type in ['rbm', 'dbn']:
#     type_dir = '/'.join([data_dir, learning_type])
#     if not os.path.isdir(type_dir):
#         os.makedirs(type_dir)


def __move_to_data_path():
    if os.getcwd() != data_dir:
        os.chdir(data_dir)


def move_to_root():
    os.chdir(root_dir)


def create_path(path):
    __move_to_data_path()
    if not os.path.isdir(path):
        os.makedirs(path)
    move_to_root()


def store_object(obj, name=None):
    if not name:
        name = str(obj)
    name += ".save"
    f = open(name, 'wb')
    Store.dump(obj, f, protocol=Store.HIGHEST_PROTOCOL)
    f.close()


def retrieve_object(name):
    filename = name + ".save"
    if os.path.isfile(filename):
        f = open(filename, 'rb')
        obj = Store.load(f)
        f.close()
        return obj
    else:
        return None


def move_to(out_dir):
    if not os.path.isabs(out_dir):
        out_dir = '/'.join([data_dir, out_dir])

    if not os.path.isdir(out_dir):
        create_path(out_dir)

    os.chdir(out_dir)
    return os.getcwd()


def store_object_to_dir(obj, dir, name=None):
    move_to(dir)
    store_object()
    move_to_root()


def retrieve_object_from_dir(out_dir, name):
    move_to(out_dir)
    obj = retrieve_object(name)
    move_to_root()
    return obj