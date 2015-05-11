__author__ = 'joschlemper'

import cPickle as Store
import os.path

'''
Storage Helper

offers two interface: store and retrieve

'''


# Path
root_dir = os.path.split(__file__)[0]
DATA_DIR = "data"
data_dir = "/".join([root_dir, DATA_DIR])


class StorageManager(object):
    def __init__(self, project_root='.', log=True):
        self.project_root = project_root
        self.log = True
        if log:
            print '... data manager created. project_root: {}'.format(project_root)
        self.move_to_project_root()

    def move_to(self, out_dir):
        move_to('/'.join([self.project_root, out_dir]))
        print '... moved to {}'.format(os.getcwd())


    def move_to_project_root(self):
        move_to(self.project_root)
        print '... moved to {}'.format(os.getcwd())

    def persist(self, obj, name=None, out_dir=None):
        cur = os.getcwd()
        if out_dir:
            self.move_to(out_dir)
        store_object(obj, name)
        print '... saved {} to {}'.format(name, os.getcwd())
        os.chdir(cur)

    def retrieve(self, name, out_dir=None):
        cur = os.getcwd()
        if out_dir:
            self.move_to(out_dir)
        obj = retrieve_object(name)
        print '... retrieved {} from {}'.format(name, os.getcwd())
        os.chdir(cur)
        return obj

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