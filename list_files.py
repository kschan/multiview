import glob
import os
import argparse


parser = argparse.ArgumentParser(description='list kitti files')
parser.add_argument('--num_views', type=int, help='number of views', default=1)
parser.add_argument('--data_path', type=str, help='path to the kitti dataset', required=True)
args = parser.parse_args()

num_views = args.num_views
out = open(args.data_path + "/filenames.txt", "w")

dirs = os.walk(args.data_path)
for (long_root, dirs, files) in dirs:
	root = long_root[len(args.data_path):]
	if root[-13:-8] != 'image':		# ignore the odometry directories
		continue
	if dirs == []:	# bottom level directory with images
		sorted_files = sorted(files)
		if (len(files) - 1) != int(sorted_files[-1][:-4]):
			print "missing files in ", root
			break
		else:
			print root, "ok"

		if root[-13:-5] == 'image_03':
			image_02_path = root[:-13] + 'image_02/data'
			for i in range(num_views-1,len(sorted_files)):
				right_views = ['/'.join([root, sorted_files[i - j]]) for j in range(num_views)]
				left_views = ['/'.join([image_02_path, sorted_files[i - j]]) for j in range(num_views)]

				out.write(' '.join(left_views + right_views) + '\n')

out.close()
