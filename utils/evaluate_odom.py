import tensorflow as tf
import numpy as np
from scipy.misc import imread
import argparse


parser = argparse.ArgumentParser(description='odometry estimate evaluator')
parser.add_argument('--data_path',                 type=str,   help='path to the kitti dataset', required=True)
parser.add_argument('--training_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--validation_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--log_directory',             type=str,   help='directory containing checkpoints and summaries', required=True)

args = parser.parse_args()

def read_image(image_path):
    image = imread(image_path)
    return image

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def image_path_to_odom_path(image_path):
    line = image_path.split('/')
    line[2] = 'oxts'
    line[-1] = line[-1][:-3] + 'txt'
    odom_file = '/'.join(line)

    return odom_file

def evaluate(sess, graph, filename):
    
    model_input = graph.get_tensor_by_name('shuffle_batch:0') # <tf.Tensor 'shuffle_batch:0' shape=(8, 256, 512, 6) dtype=float32>
    odom_logits = [graph.get_tensor_by_name(x + '/egomotion/odom_prediction/BiasAdd:0') for x in ['model', 'model_1', 'model_2', 'model_3']]
    odom_prediction = tf.nn.softmax(tf.concat(odom_logits, 0))
    # odom_prediction = tf.concat(odom_logits, 0)
    
    batch_size = int(model_input.get_shape()[0])
    num_correct = 0
    num_correct_only_turn = 0
    total = 0

    with open(filename, 'r') as validation_examples:
        for _ in xrange(count_text_lines(filename)//batch_size):
            inputs = []
            odom_labels = []
            for i in xrange(batch_size):
                line = validation_examples.readline().split()
                first = read_image(args.data_path + line[0])
                second = read_image(args.data_path + line[1])

                odom_path = args.data_path + image_path_to_odom_path(line[0])
                with open(odom_path, 'r') as oxts_file:
                    oxts = [int(float(x)) for x in oxts_file.readline().split()]
                
                vf = oxts[8]    # FORWARD VELOCITY [m/s]
                vl = oxts[9]    # LEFTWARDS VELOCITY
                
                change_order = np.random.uniform() > 0.5
                if change_order:
                    example = np.concatenate([second, first], 2)
                    vf *= -1
                    vl *= -1
                else:
                    example = np.concatenate([first, second], 2)
                
                inputs.append(example)

                angles = np.arctan2(vf, vl) * 180/np.pi
                angles = (angles + 360.)%360.

                binned_angles = int(angles > 180)
                odom_labels.append(binned_angles)

            outputs = sess.run(odom_prediction, feed_dict={model_input:inputs}) # I expect shape [8, 8]
            output_classes = np.argmax(outputs, axis = 1)
            print output_classes
            num_correct += np.sum(odom_labels == output_classes)
            total += len(output_classes)
    return num_correct / float(total)

def main():
    config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
    sess=tf.Session(config=config)
    #First let's load meta graph and restore weights

    saver = tf.train.import_meta_graph(tf.train.latest_checkpoint(args.log_directory) + '.meta', clear_devices = True)
    saver.restore(sess,tf.train.latest_checkpoint(args.log_directory))

    graph = tf.get_default_graph()
    train_accuracy = evaluate(sess, graph, args.training_file)
    validation_accuracy = evaluate(sess, graph, args.validation_file) 
    sess.close()

    with open(args.log_directory + '/validation_accuracies.txt', 'a') as output:
        step = tf.train.latest_checkpoint(args.log_directory).split('-')[-1]
        output.write('%s, %f, %f\n' % (step, train_accuracy, validation_accuracy))

if __name__ == '__main__':
    main()
