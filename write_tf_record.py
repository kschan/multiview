import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import sys
import os


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

kitti = "../kitti"
lefts  = glob.glob(kitti + "/*/*/image_02/data/*.jpg")
rights = glob.glob(kitti + "/*/*/image_03/data/*.jpg")

# confirm not corrupted or missing files

if len(lefts) != len(rights):
    print "missing files"
    sys.exit()

print len(lefts), "examples"

for i in range(len(lefts)):
    if lefts[i][-14:] != rights[i][-14:]:
        print "misaligned files"
        sys.exit()

print "all ok!"

tfrecord_filename = kitti + "/kitti.tfrecords"
writer = tf.python_io.TFRecordWriter(tfrecord_filename)


for i in range(len(lefts)):
    left_img  = mpimg.imread(lefts[i]).tostring()
    right_img = mpimg.imread(rights[i]).tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'left': _bytes_feature(left_img),
        'right': _bytes_feature(right_img)}))

    writer.write(example.SerializeToString())

writer.close()

# record_iterator = tf.python_io.tf_record_iterator(path=kitti + "/kitti.tfrecords")
# for string_record in record_iterator:
#     example = tf.train.Example()

#     example.ParseFromString(string_record)
    
#     left_string = (example.features.feature['left']
#                                   .bytes_list
#                                   .value[0])
    
#     right_string = (example.features.feature['right']
#                                 .bytes_list
#                                 .value[0])

#     left_img = np.fromstring(left_string, dtype=np.uint8).reshape((256, 512, -1))
#     right_img = np.fromstring(right_string, dtype=np.uint8).reshape((256, 512, -1))

#     plt.imshow(left_img)
#     plt.title("left img")
#     plt.imshow(right_img)
#     plt.title("right_img")
#     plt.show()