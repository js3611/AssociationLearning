__author__ = 'joschlemper'

import cPickle
import gzip
import os
import sys
import time
# import cv2
from sklearn import preprocessing
from sklearn import cross_validation
import numpy as np
import theano
import theano.tensor as T

try:
    import PIL.Image as Image
except ImportError:
    import Image

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')

emotion_dict = {'anger': 1, 'contempt': 2, 'disgust': 3, 'fear': 4, 'happy': 5, 'sadness': 6, 'surprise': 7}
emotion_rev_dict = {1: 'anger', 2: 'contempt', 3: 'disgust', 4: 'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'}


def load_kanade(shared=True, set_name='sharp_equi25_25', emotions=None, pre=None, n=None):
    '''
    :param shared: To return theano shared variable. If false, returns in numpy
    :param digits: filter. if none, all digits will be returned
    :param pre: a dictionary of pre-processing. Options: pca, white-pca, center, threshold
    :param n: (a single digit or) an array showing how many samples to get
    :return: [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    '''

    data = __load(set_name)

    # Shuffle data
    idx = np.random.choice(range(0, len(data[1])), size=len(data[1]), replace=False)
    data = (data[0][idx], data[1][idx])

    # pre-processing
    if emotions:  #filter
        x, y = data
        filter_keys = map(lambda x: emotion_dict[x], emotions)
        filtered = filter(lambda (x, y): y in filter_keys, enumerate(y))
        idx = [s[0] for s in filtered]
        data = (x[idx], y[idx])

    if n:
        idx = np.random.randint(0, len(data[1]), size=n)
        data = (data[0][idx], data[1][idx])

    if pre:
        if 'pca' in pre:
            pass
        if 'wpca' in pre:
            pass
        if 'scale2unit' in pre:
            x = data[0] / 255.0
            data = (x, data[1])
        if 'scale' in pre:
            x = preprocessing.scale(data[0].astype(np.float32))
            data = (x, data[1])
        if 'center' in pre:
            pass
        if 'threshold' in pre:
            data = get_binary(data, pre['threshold'])
        if 'binary_label' in pre:
            data = get_binary_label(data)
        if 'label_vector' in pre:
            data = vectorise_label(data)

    # split to train and test
    rand = 123
    tr_te = cross_validation.train_test_split(data[0], data[1], test_size=(2.0/7), random_state=rand)
    tr_x, te_x, tr_y, te_y = tr_te
    vl_x, te_x, vl_y, te_y = cross_validation.train_test_split(te_x, te_y, test_size=0.5, random_state=rand)

    data = [(tr_x, tr_y), (vl_x, vl_y), (te_x, te_y)]

    if shared:
        data = get_shared(data)
        # data = shared_dataset(data)

    return data


def __load(set_name='25_25'):
    ''' Loads the mnist data set '''
    print BASE_DIR

    dataset_name = 'kanade' + set_name + '.save'

    # look for the location
    possible_locations = ['', 'data']
    dataset = DATA_DIR
    for location in possible_locations:
        data_location = os.path.join(BASE_DIR, location, dataset_name)
        if os.path.isfile(data_location):
            print '... dataset found at {}'.format(data_location)
            dataset = data_location
            break

    # If the saved file doesn't exist, create one
    # if not os.path.isfile(data_location):
    #     dir_name = 'kanade/' + set_name
    #     for location in possible_locations:
    #         data_location = os.path.join(BASE_DIR, location, dir_name)
    #         if os.path.isdir(data_location):
    #             dataset = data_location
    #             break
    #
    #     assert os.path.isdir(dataset)
    #     print '... creating data'
    #
    #     data = []
    #     label = []
    #     for emotion in emotion_dict.keys():
    #         for img in os.listdir(dataset + '/' + emotion):
    #             img_name = os.path.join(dataset, emotion, img)
    #             img_array = cv2.imread(img_name, 0)
    #             data.append(np.asarray(img_array).reshape(-1))
    #             label.append(emotion_dict[emotion])
    #
    #     dataset = os.path.join(DATA_DIR, 'kanade' + set_name + '.save')
    #
    #     # print data
    #     # print label
    #     # print os.getcwd()
    #
    #     f = open(dataset, 'wb')
    #     cPickle.dump((np.array(data), np.array(label)), f, protocol=cPickle.HIGHEST_PROTOCOL)
    #     f.close()

    # open
    f = open(dataset, 'rb')
    data, label = cPickle.load(f)
    f.close()

    return data, label



def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch every time
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def get_shared(data):
    train_set, valid_set, test_set = data
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)
    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


def get_binary(data, t=0.5):
    # Preprocessing
    new_data = []
    for data_xy in data:
        new_data.append((to_binary(data_xy[0]), data_xy[1]))
    return new_data


def to_binary(data, t=0.5):
    """
    :param data: 2 dimensional np array
    :param t: threshold value
    :return: data with binary data, 1 if data[i] > t, 0 otherwise
    """
    data[data >= t] = 1
    data[data < t] = 0
    return data


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def get_binary_label(data):
    new_data = []
    for (x, y) in data:
        new_data.append((x, (y % 2)))
    return new_data


def get_target_vector(x):
    xs = np.zeros(10, dtype=theano.config.floatX)
    xs[x] = 1
    return xs


def sample_image(data, shared=True, mapping=None):
    # convert to numpy first
    if 'Tensor' in str(type(data)):
        seq = data.eval()
    else:
        seq = data

    if not mapping:
        mapping = {}
        for emo in emotion_dict.keys():
            mapping[emo] = emo  # get id map

    source_emotions = np.unique(seq).tolist()
    target_emotions = list(set(map(lambda x: mapping[emotion_rev_dict[x]], source_emotions)))
    image_pool = {}
    for d in target_emotions:
        dataset = load_kanade(shared=False, set_name='sharp_equi25_25', emotions=[d], n=len(seq), pre={'scale':True})
        image_pool[d] = dataset[0][0]

    sample_data = []
    sample_label = []
    rand_seq = np.random.randint(0, len(seq), size=len(seq))

    for d, r in zip(seq.tolist(), rand_seq.tolist()):
        pool = image_pool[mapping[emotion_rev_dict[d]]]
        sample_data.append(pool[r % len(pool)])
        sample_label.append(emotion_dict[mapping[emotion_rev_dict[d]]] * 1.)

    if shared:
        data = theano.shared(np.array(sample_data, dtype=theano.config.floatX), borrow=True)
        label = theano.shared(np.array(sample_label, dtype=theano.config.floatX), borrow=True)
        return data, T.cast(label, 'int32')
    else:
        return np.asarray(sample_data, dtype=theano.config.floatX), np.asarray(sample_label, dtype=theano.config.floatX)

def vectorise_label(data):
    new_data = []
    for (x, y) in data:
        new_data.append((x, np.array(map(get_target_vector, y))))
    return new_data


def load_shared():
    train_set, valid_set, test_set = __load()

    # Convert to theano shared variables
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_data_threshold(dataset, t=0.5):
    [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)] = __load()
    new_train_x = to_binary(train_set_x, t)
    new_valid_x = to_binary(valid_set_x, t)
    new_test_x = to_binary(test_set_x, t)

    return [(new_train_x, train_set_y), (new_valid_x, valid_set_y),
            (new_test_x, test_set_y)]


def save_faces(x, shape=None, image_name='digits.png', img_shape=(25,25)):
    data_size = x.shape[0]
    nrow, ncol = img_shape
    image_data = np.zeros(((nrow+1), (ncol+1) * data_size - 1), dtype='uint8')
    if not shape:
        shape = (1, data_size)

    image_data = tile_raster_images(
        X=x,
        img_shape=img_shape,
        tile_shape=shape,
        tile_spacing=(1, 1)
    )

    # construct image
    image = Image.fromarray(image_data)
    image.save(image_name)


def save_face(x, name="digit.png", img_shape=(25,25)):
    image_data = np.zeros(img_shape, dtype='uint8')

    # Original images
    image_data = tile_raster_images(
        X=np.array([x]),
        img_shape=img_shape,
        tile_shape=(1, 1),
        tile_spacing=(1, 1)
    )

    image = Image.fromarray(image_data)
    image.save(name)


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array
