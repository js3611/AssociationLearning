__author__ = 'joschlemper'

import numpy as np
import utils
import PIL.Image as Image
import sys
import time


class ProgressLogger(object):

        def __init__(self,
                     likelihood=True,
                     monitor_weights=True,
                     time_training=True,
                     plot=True,
                     plot_info=None,
                     img_shape=(28, 28),
                     out_dir=None
                     ):

            self.likelihood = likelihood
            self.time_training = time_training
            self.plot = plot
            self.plot_info = plot_info
            self.out_dir = out_dir
            self.monitor_weights = monitor_weights
            self.weight_hist = {'avg': [], 'std': [], 'min': sys.maxint, 'max': -sys.maxint - 1}
            self.img_shape = img_shape

        def visualise_weight(self, rbm, image_name):
            plotting_start = time.clock()  # Measure plotting time

            if rbm.v_n in [784, 625, 1250, 784*2, 2500, 5000]:
                tile_shape = (rbm.h_n / 10 + 1, 10)

                image = Image.fromarray(
                    utils.tile_raster_images(
                        X=rbm.W.get_value(borrow=True).T,
                        img_shape=self.img_shape,
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                )
                image.save(image_name)

            plotting_end = time.clock()
            return plotting_end - plotting_start

        def visualise_reconstructions(self, orig, reconstructions, plot_n=None, img_name='reconstruction', opt=False):
            if opt:
                img_shape = (self.img_shape[0]/2, self.img_shape[1])
            else:
                img_shape = self.img_shape

            visualise_reconstructions(orig, reconstructions, img_shape, plot_n, img_name)

        def monitor_wt(self, rbm):
            # print rbm.W.get_value(borrow=True)[0][0:5]
            self.weight_hist['avg'].append(np.mean(rbm.W.get_value(borrow=True)))
            self.weight_hist['std'].append(np.std(rbm.W.get_value(borrow=True)))
            self.weight_hist['min'] = min(np.min(rbm.W.get_value(borrow=True)), self.weight_hist['min'])
            self.weight_hist['max'] = max(np.max(rbm.W.get_value(borrow=True)), self.weight_hist['max'])

        def monitor_mean_activity(self, rbm, x, y):
            _, hp = rbm.prop_up(x, y)
            pass


class AssociationProgressLogger(ProgressLogger):
    def visualise_weight(self, rbm, image_name):
            assert rbm.associative
            if rbm.v_n in [784, 625, 2500, 5000]:
                plotting_start = time.clock()  # Measure plotting time

                w = rbm.W.get_value(borrow=True).T
                u = rbm.U.get_value(borrow=True).T

                weight = np.hstack((w, u))

                tile_shape = (rbm.h_n / 10 + 1, 10)

                image = Image.fromarray(
                    utils.tile_raster_images(
                        X=weight,
                        img_shape=(self.img_shape[0] *2, self.img_shape[1]),
                        tile_shape=tile_shape,
                        tile_spacing=(1, 1)
                    )
                )
                image.save(image_name)

                plotting_end = time.clock()
                return plotting_end - plotting_start
            return 0

    def visualise_reconstructions(self, orig, reconstructions, plot_n=None, img_name='association'):
        visualise_reconstructions(orig, reconstructions, self.img_shape, plot_n, img_name)


def visualise_reconstructions(orig, reconstructions, img_shape, plot_n=None, img_name='reconstruction'):
            k = len(reconstructions)
            assert k > 0

            if plot_n:
                data_size = min(plot_n, orig.shape[0])
            else:
                data_size = orig.shape[0]

            if orig.shape[1] in [784, 625, 1250, 784*2, 2500, 5000]:
                nrow, ncol = img_shape

                image_data = np.zeros(
                    ((nrow+1) * (k+1) + 1, (ncol+1) * data_size - 1),
                    dtype='uint8'
                )

                # Original images
                image_data[0:nrow, :] = utils.tile_raster_images(
                    X=orig,
                    img_shape=img_shape,
                    tile_shape=(1, data_size),
                    tile_spacing=(1, 1)
                )

                # Generate image by plotting the sample from the chain
                for i in xrange(1, k+1):
                    # print ' ... plotting sample ', i
                    idx = (nrow+1) * i
                    image_data[idx:(idx+nrow), :] = utils.tile_raster_images(
                        X=(reconstructions[i-1]),
                        img_shape=img_shape,
                        tile_shape=(1, data_size),
                        tile_spacing=(1, 1)
                    )

                # construct image
                image = Image.fromarray(image_data)
                image.save(img_name + '_{}.png'.format(k))