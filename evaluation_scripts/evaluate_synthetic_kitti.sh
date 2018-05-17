mkdir -p $HOME/disp_results/synthetic_kitti
python monodepth_main.py --mode test --encoder $2 --data_path /data/synthetic_kitti/frames_cleanpass_webp/35mm_focallength/scene_forwards/fast/ --filenames_file /data/synthetic_kitti/frames_cleanpass_webp/35mm_focallength/scene_forwards/fast/filenames.txt --log_directory $HOME/logs/ --model_name $1 --output_directory $HOME/disp_results/synthetic_kitti --num_gpus 1
 
python utils/evaluate_flyingthings.py --predicted_disp_path $HOME/disp_results/synthetic_kitti/$1/disparities.npy --gt_path /data/synthetic_kitti/disparity/35mm_focallength/scene_forwards/fast/ --filenames_file /data/synthetic_kitti/frames_cleanpass_webp/35mm_focallength/scene_forwards/fast/filenames.txt
