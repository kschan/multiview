import numpy as np
import cv2
import argparse
import re
from evaluation_utils import *
from skimage.transform import resize
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluation on the Kevin dataset')

parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--data_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--filenames_file',      type=str,   help='filenames_file',                     required=True)

args = parser.parse_args()

if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)
    pred_disparities = np.array([resize(disp, (240, 320)) * 320 for disp in pred_disparities])
    num_samples = count_lines(args.filenames_file)

    print pred_disparities.shape 
    pred_depths = 160.5 * 0.54/pred_disparities
    pred_depths[pred_depths>50] = 50
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    scale_inv = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    with open(args.filenames_file, 'r') as f:
        for i in range(num_samples):
            splits = re.split('/|\.| |', f.readline())
            dataset = splits[0]
            file_num = splits[2]
            gt_file = args.data_path + dataset + '/depth/' + file_num + '.npy'
            gt_depth = np.load(gt_file)
            pred_depth = pred_depths[i]
             
            # pred_depth = pred_depth[:, 10:]
            # gt_depth = gt_depth[:, 10:]
            
            # pred_depth = pred_depth * 0.0763/0.7

            ''' 
            print np.max(pred_depth)            
            plt.subplot(211)
            plt.imshow(gt_depth)
            plt.subplot(212)
            plt.imshow(pred_depth) 
            plt.show()
            '''

            mask = gt_depth > 0
            
            abs_rel[i], sq_rel[i], rms[i], log_rms[i], scale_inv[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'scale_inv', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:11.4f}, {:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), scale_inv.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))
