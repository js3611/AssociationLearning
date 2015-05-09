import unittest
import datastorage as store
import utils
import rbm as RBM
import DBN as DBN

import os


class UtilsTest(unittest.TestCase):
    def test_storage(self):
        path = "association_test_2"
        store.create_path(path)
        self.assertTrue(os.path.isdir('/'.join([store.data_dir, path])))

        path = "association_test_3"
        store.move_to(path)
        self.assertTrue(os.getcwd() == '/'.join([store.data_dir, path]))

        store.move_to_root()
        self.assertTrue(os.getcwd() == store.root_dir)

        store.move_to(path)
        rbm = RBM.RBM()
        store.store_object(rbm)

        rbm2 = store.retrieve_object(str(rbm))

        self.assertTrue(str(rbm) == str(rbm2))

        dbn = DBN.DBN()
        store.store_object(dbn)

        dbn2 = store.retrieve_object(str(dbn))
        self.assertTrue(str(dbn) == str(dbn2))





        store.move_to_root()


if __name__ == '__main__':
    print "Test Utilities"
    unittest.main()