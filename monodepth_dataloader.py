# Copyright UCL Business plc 2017. Patent Pending. All ri1ghts reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Monodepth data loader.
"""

import tensorflow as tf
import numpy as np
from constants import *

def string_length_tf(t):
  return tf.py_func(len, [t], [tf.int64])

class MonodepthDataloader(object):
    """monodepth dataloader"""

    def __init__(self, data_path, filenames_file, params, dataset, mode):
        self.data_path = data_path
        self.params = params
        self.dataset = dataset
        self.mode = mode
        self.left_image_batch  = None
        self.right_image_batch = None

        input_queue = tf.train.string_input_producer([filenames_file], shuffle=True)
        line_reader = tf.TextLineReader()
        _, line = line_reader.read(input_queue)

        split_line = tf.string_split([line]).values

        # we load only one image for test, except if we trained a stereo model
        # if mode == 'test' and not self.params.do_stereo:
        #     left_image_path  = tf.string_join([self.data_path, split_line[0]])
        #     left_image_o  = self.read_image(left_image_path)
        # else:

        self.first_image_path = tf.string_join([self.data_path, split_line[0]])
        left_image_o_1 = self.read_image(tf.string_join([self.data_path, split_line[0]]))
        left_image_o_2 = self.read_image(tf.string_join([self.data_path, split_line[1]]))
        right_image_o_1 = self.read_image(tf.string_join([self.data_path, split_line[2]]))
        right_image_o_2 = self.read_image(tf.string_join([self.data_path, split_line[3]]))


        path = tf.string_split([split_line[0]], delimiter='/').values
        self.oxts_path = tf.string_join([self.data_path, path[0], '/', path[1], '/oxts/data/', tf.string_split([path[-1]], delimiter='.').values[0], '.txt'])
        all_oxts_strings = tf.read_file(self.oxts_path)
        all_oxts_strings = tf.string_split([all_oxts_strings], delimiter=' ')
        all_oxts_strings = tf.sparse_tensor_to_dense(all_oxts_strings, default_value='0')
        all_oxts = tf.string_to_number(all_oxts_strings, out_type=tf.float64)
        all_oxts = tf.reshape(all_oxts, [-1])

        # Velocities
        # vf_o = oxts[8]    # FORWARD VELOCITY [m/s]
        # vl_o = oxts[9]    # LEFTWARDS VELOCITY
        # vu_o = oxts[10]   # UP VELOCITY

        # wf_o = oxts[20]   # FORWARD AXIS ROTATION
        # wl_o = oxts[21]   # LEFTWARDS AXIS ROTATION
        # wu_o = oxts[22]   # UPWARDS AXIS ROTATION

        oxts_o = tf.concat([all_oxts[8:11], all_oxts[20:23]], axis=0) / \
                    np.array([VF_VARIANCE, VL_VARIANCE, VU_VARIANCE, WF_VARIANCE, WL_VARIANCE, WU_VARIANCE])**0.5
        # normalize oxts according to parameters in constants.py

        if mode == 'train':
            # we want to fight the bias on stationary images in the dataset, so occasionally
            # we feed the network the same two images to represent no flow information.
            repeat_images = tf.random_uniform([], 0, 1)
            repeat_threshold = 0.9
            right_image_o_2 = tf.cond(repeat_images > repeat_threshold,
                                        lambda: right_image_o_1,
                                        lambda: right_image_o_2
                              )
            left_image_o_2 = tf.cond(repeat_images > repeat_threshold,
                                        lambda: left_image_o_1,
                                        lambda: left_image_o_2
                              )
            oxts_o         = tf.cond(repeat_images > repeat_threshold,
                                        lambda: oxts_o * 0.0,
                                        lambda: oxts_o
                              )
            
            # we randomly change the order of the images to teach it to do sfm generally
            change_order = tf.random_uniform([], 0, 1)  # if this is > 0.5, swap the images so the second image is first 3 layers
            change_order_threshold = 0.5
            right_image_o = tf.cond(change_order > change_order_threshold,
                                lambda: tf.concat([right_image_o_2, right_image_o_1], axis = 2),
                                lambda: tf.concat([right_image_o_1, right_image_o_2], axis = 2)
                            )
            left_image_o = tf.cond(change_order > change_order_threshold,
                                lambda: tf.concat([left_image_o_2, left_image_o_1], axis = 2),
                                lambda: tf.concat([left_image_o_1, left_image_o_2], axis = 2)
                            )
         
            # TODO: This is a bit dodgy because if we flip the order of the images, we are using the odometry informatinon for the (now) second image
            oxts = tf.cond(change_order > change_order_threshold, lambda: -oxts_o, lambda: oxts_o) 
            # randomly flip images
            do_flip = tf.random_uniform([], 0, 1)
            threshold = 0.5
            left_image  = tf.cond(do_flip > threshold, lambda: tf.image.flip_left_right(right_image_o), lambda: left_image_o)
            right_image = tf.cond(do_flip > threshold, lambda: tf.image.flip_left_right(left_image_o),  lambda: right_image_o)
            
            # If we flip the images left to right, we need to negate horizontal velocity and rotation
            oxts = tf.cond(do_flip > threshold, lambda: oxts * np.array([1, -1, 1, 1, -1, 1]), lambda: oxts)
            
            # randomly augment images
            do_augment  = tf.random_uniform([], 0, 1)
            left_image, right_image = tf.cond(do_augment > 0.5, lambda: self.augment_image_pair(left_image, right_image), lambda: (left_image, right_image))

            left_image.set_shape( [None, None, 6])
            right_image.set_shape([None, None, 6])

            # capacity = min_after_dequeue + (num_threads + a small safety margin) * batch_size
            min_after_dequeue = 2048
            capacity = min_after_dequeue + (params.num_threads + 4) * params.batch_size
            self.left_image_batch, self.right_image_batch, self.oxts_batch = tf.train.shuffle_batch([left_image, right_image, oxts],
                        params.batch_size, capacity, min_after_dequeue, params.num_threads)

        elif mode == 'test':
            left_image_o = tf.concat([left_image_o_1, left_image_o_2], axis = 2)
            left_image = tf.stack([left_image_o,  tf.image.flip_left_right(left_image_o)],  0)
            left_image.set_shape( [2, None, None, 6])
            
            self.left_image_batch = left_image

    def augment_image_pair(self, left_image, right_image):
        left_image_aug = left_image
        right_image_aug = right_image

        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_image_aug  = left_image  ** random_gamma
        right_image_aug = right_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_image_aug  =  left_image_aug * random_brightness
        right_image_aug = right_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([6], 0.8, 1.2)
        white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(6)], axis=2)
        left_image_aug  *= color_image
        right_image_aug *= color_image

        # saturate
        left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
        right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

        return left_image_aug, right_image_aug

    def read_image(self, image_path):
        # tf.decode_image does not return the image size, this is an ugly workaround to handle both jpeg and png
        path_length = string_length_tf(image_path)[0]
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')
        
        image  = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)), lambda: tf.image.decode_png(tf.read_file(image_path)))

        # if the dataset is cityscapes, we crop the last fifth to remove the car hood
        if self.dataset == 'cityscapes':
            o_height    = tf.shape(image)[0]
            crop_height = (o_height * 4) / 5
            image  =  image[:crop_height,:,:]

        image  = tf.image.convert_image_dtype(image,  tf.float32)
        
        if self.mode != 'train':
            image  = tf.image.resize_images(image,  [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)
        else:
            image.set_shape([256, 512, 3])
        return image
