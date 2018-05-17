import numpy as np
import cv2
import argparse
from evaluation_utils import *

parser = argparse.ArgumentParser(description='Evaluation on the Kevin dataset')

parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--filenames_file',      type=str,   help='filenames_file',                     required=True)

args = parser.parse_args()

if __name__ == '__main__':

    pred_disparities = np.load(args.predicted_disp_path)

    num_samples = count_lines(args.filenames_file)
    gt_depths = np.load(args.gt_path) 
    pred_depths = 718.3 * 0.54/pred_disparities

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        gt_disp = gt_disparities[i]
        mask = gt_disp > 0
        
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))
