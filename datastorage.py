__author__ = 'joschlemper'

import cPickle as Store


def store_object(obj, name=None):
    if not name:
        name = str(obj)
    name += ".save"
    f = open(name, 'wb')
    Store.dump(obj, f, protocol=Store.HIGHEST_PROTOCOL)
    f.close()


def retrieve_object(name):
    filename = name + ".save"
    f = open(filename, 'rb')
    obj = Store.load(f)
    f.close()
    return obj

