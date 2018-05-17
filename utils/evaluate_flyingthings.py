import numpy as np
import cv2
from evaluation_utils import compute_errors
import argparse
import re
import sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Evaluation on the Flyingthings3d dataset')
parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
parser.add_argument('--filenames_file',      type=str,   help='what files did you generate disparities for', required=True)
parser.add_argument('--train_dataset',       type=str,   required=True)
args = parser.parse_args()

def countlines(file):
    f = open(file, 'r')
    res = len(f.readlines())
    f.close()
    return res

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def get_gt_disp(file):
    data, scale = readPFM(file)
    return data

def flying_disp_to_depth(disp):
    # 1050 pixel focal length, baseline is one blender unit which they claim is 
    # proportional to kitti baseline of 0.54 m, so use KITTI baseline here.
    
    baseline = 0.93 # meters 
    return 1050 * baseline / disp

def kitti_disp_to_depth(disp):
    return 718 * 0.54 / disp 

def get_gt_disps_depths(args):
    gt_depths = []
    gt_disps = []
    num_examples = countlines(args.filenames_file)
    with open(args.filenames_file, 'r') as f:
        for i in range(num_examples):
            left_img_path = f.readline().split()[0]
            disp_path = args.gt_path + left_img_path[:-3] + 'pfm'
            disp = get_gt_disp(disp_path)   # pixels
            gt_disps.append(disp)
            
            gt_depth = flying_disp_to_depth(disp)  # meters
            gt_depths.append(gt_depth)
    
    return gt_disps, gt_depths

def get_pred_disp_resized_depth(disp):
    width, height = 960, 540     # gt size
    
    # plt.imshow(disp); plt.show()
    pred_resized = width * cv2.resize(disp, (width, height), interpolation=cv2.INTER_LINEAR)    # pixels
    
    # plt.imshow(pred_resized); plt.show()
    if args.train_dataset == 'kitti':
        return pred_resized, kitti_disp_to_depth(pred_resized)
    elif args.train_dataset == 'flying':
        return pred_resized, flying_disp_to_depth(pred_resized)
    else:
        print "invalid dataset"
        sys.exit(0)

def main():
    num_samples = countlines(args.filenames_file)

    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    scale_inv = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    pred_disps = np.load(args.predicted_disp_path)      # disparity as fraction of image width
    gt_disps, gt_depths = get_gt_disps_depths(args)
    # np.save('synthetic_kitti_disps', gt_disps )
    print "starting:"
    for i in range(num_samples):
        # print i + '/' + num_samples
        pred_disp = pred_disps[i]
        pred_disp, pred_depth = get_pred_disp_resized_depth(pred_disp)
        gt_disp = gt_disps[i]
        gt_depth = gt_depths[i]
        # plt.subplot(211);plt.imshow(gt_depth); plt.subplot(212); plt.imshow(pred_depth);plt.show()
        disp_diff = np.abs(gt_disp - pred_disp)
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / (960*540.)
        mask = gt_depth < 50
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], scale_inv[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
        #print abs_rel[i]
    print abs_rel
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'scale_inv', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), scale_inv.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))
'''
    with open(args.predicted_disp_path + '/flyingthings3d_result.txt', 'w') as f:
        f.write('%s\n \
                 %s' % (header_string, result_string))
'''

if __name__ == '__main__':
    main()
