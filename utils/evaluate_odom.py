import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='odometry estimate evaluator')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')


sess=tf.Session()
#First let's load meta graph and restore weights

saver = tf.train.import_meta_graph(parser.log_directory + parser.model_name + '.meta')
saver.restore(sess,tf.train.latest_checkpoint(parser.log_directory))


# Access saved Variables directly
print(sess.run('bias:0'))
# This will print 2, which is the value of bias that we saved


# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

input = graph.get_tensor_by_name('split:0') # <tf.Tensor 'split:0' shape=(8, 256, 512, 6) dtype=float32>    

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print sess.run(op_to_restore,feed_dict)
#This will print 60 which is calculated 