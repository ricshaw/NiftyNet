# -*- coding: utf-8 -*-
"""
windows aggregator resize each item
in a batch output and save as an image
"""
from __future__ import absolute_import, print_function, division

import os

import numpy as np

import niftynet.io.misc_io as misc_io
from niftynet.engine.windows_aggregator_base import ImageWindowsAggregator
from niftynet.layer.discrete_label_normalisation import \
    DiscreteLabelNormalisationLayer
from niftynet.layer.pad import PadLayer
import tensorflow as tf


class ClassifierAggregatorMultioutput(ImageWindowsAggregator):
    """
    This class decodes each item in a batch by saving classification
    labels to a new image volume.
    """
    def __init__(self,
                 image_reader,
                 name='image',
                 output_path=os.path.join('.', 'output'),
                 prefix='_niftynet_out'):
        ImageWindowsAggregator.__init__(
            self, image_reader=image_reader, output_path=output_path)
        self.name = name
        self.image_out = None
        self.output_interp_order = 0
        self.prefix = prefix
        self.csv_path = os.path.join(self.output_path, self.prefix+'.csv')
        if os.path.exists(self.csv_path):
            os.remove(self.csv_path)

    def decode_batch(self, window, location):
        """
        window holds the classifier labels
        location is a holdover from segmentation and may be removed
        in a later refactoring, but currently hold info about the stopping
        signal from the sampler
        """
        print('\nEntered decode_batch\n')
        for w in window:
            print(w, window[w].shape, np.sum(np.isnan(window[w]),axis=(0,1,2,3)))
            n_samples = window[w].shape[0]
            print('n_samples: ', n_samples)

        for batch_id in range(n_samples):
            if self._is_stopping_signal(location[batch_id]):
                return False

            self.image_id = location[batch_id, 0]
            image_id, x_start, y_start, z_start, x_end, y_end, z_end = location[batch_id, :]
            #print(image_id, x_start, y_start, z_start, x_end, y_end, z_end)

            '''if image_id != self.image_id:
                # image name changed: save current image and create an empty image
                self._save_current_image(name_opt)
                if self._is_stopping_signal(location[batch_id]):
                    return False
                self.image_out = {}
                for w in window:
                    self.image_out[w] = self._initialise_empty_image(
                        image_id=image_id,
                        n_channels=window[w].shape[-1],
                        dtype=window[w].dtype)'''

            print('image id: ', image_id)
            # Classify label
            print('output 1: ', window['window1'][batch_id, ...].shape)
            self._save_classify_label(window['window1'][batch_id, ...])

            # Classify image
            print('output 2: ', window['window2'][batch_id, ...].shape)
            self._save_classify_image(window['window2'][batch_id,:,:,:,:])

            # Check output
            print('output 3: ', window['window3'][batch_id, ...].shape)
            #self._save_classify_check(window['window3'][batch_id, ...])


        print('\nEnd of decode_batch\n')
        return True

    def _initialise_empty_image(self, image_id, n_channels, dtype=np.float):
        self.image_id = image_id
        spatial_shape = self.input_image[self.name].shape[:3]
        output_image_shape = spatial_shape + (n_channels,)
        empty_image = np.zeros(output_image_shape, dtype=dtype)

        for layer in self.reader.preprocessors:
            if isinstance(layer, PadLayer):
                empty_image, _ = layer(empty_image)
        return empty_image


    # Classification label
    def _save_classify_label(self, image_out):
        if self.input_image is None:
            return

        print("Label: ", image_out)
        window_shape = [1, 1, 1, 1, image_out.shape[-1]]
        image_out = np.reshape(image_out, window_shape)
        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}{}_label.nii.gz".format(subject_name, self.prefix)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                image_out,
                                source_image_obj,
                                self.output_interp_order)
        with open(self.csv_path, 'a') as csv_file:
            data_str = ','.join([str(i) for i in image_out[0, 0, 0, 0, :]])
            csv_file.write(subject_name+','+data_str+'\n')
        self.log_inferred(subject_name, filename)
        return

    # Per pixel classification
    def _save_classify_image(self, image_out):
        #image_out = tf.nn.softmax(image_out)
        image_out = image_out[:,:,:,1]
        if self.input_image is None:
            return
        print("Output image shape:", image_out.shape)
        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}{}_image.nii.gz".format(subject_name, self.prefix)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                image_out,
                                source_image_obj,
                                self.output_interp_order)
        return

    # Check image output
    def _save_classify_check(self, image_out):
        if self.input_image is None:
            return

        for layer in reversed(self.reader.preprocessors):
            if isinstance(layer, DiscreteLabelNormalisationLayer):
                image_out, _ = layer.inverse_op(image_out)
        subject_name = self.reader.get_subject_id(self.image_id)
        filename = "{}{}_check.nii.gz".format(subject_name, self.prefix)
        source_image_obj = self.input_image[self.name]
        misc_io.save_data_array(self.output_path,
                                filename,
                                image_out,
                                source_image_obj,
                                self.output_interp_order)
        return
