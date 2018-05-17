# Multiview Monocular Depth Estimation Using Unsupervised Learning Methods

To train:
`python monodepth_main.py --model_name whatever_name --encoder multiview --data_path /path/to/data/ --filenames_file /path/to/data/filenames.txt --log_directory /where/you/want/saved/models/`

The filenames_file must specify the relative path of the training examples from data_path. For training multiview, the filenames_file contains a single line for each training example. A training example consists of four images. The first two paths specified on a line are two consecutive views from the left camera of a stereo video. The next two paths specified are the corresponding right images from the stereo video.

Evaluation can be accomplished with the included shell scripts (evaluate_kitti.sh, etc.). For example, `sh evaluate_kitti.sh multiview_3 multiview` evaluates a model called multiview_3 (found in the log_directory) that has architecture multiview on the kitti evaluation set. Disparities from evaluation are saved in the output_directory specified in the shell script.
