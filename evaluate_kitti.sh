mkdir -p $HOME/disp_results/kitti
python monodepth_main.py --mode test --encoder $2 --data_path /data/kitti_resized/training/ --filenames_file /data/kitti_resized/training/filenames_2_view.txt --log_directory $HOME/logs/ --model_name $1 --output_directory $HOME/disp_results/kitti --num_gpus 1 
python utils/evaluate_kitti.py --split kitti --predicted_disp_path $HOME/disp_results/kitti/$1/disparities.npy --gt_path /data/kitti_resized/
