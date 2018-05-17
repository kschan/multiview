mkdir -p $HOME/disp_results/flyingthings3d 

python monodepth_main.py --encoder $2 --mode test --data_path /data/flyingthings3d/frames_cleanpass_webp/test_resized/ --filenames_file /data/flyingthings3d/frames_cleanpass_webp/test_resized/flying_eval_a.txt --log_directory $HOME/logs/ --model_name $1 --output_directory $HOME/disp_results/flyingthings3d --num_gpus 1  
python utils/evaluate_flyingthings.py --predicted_disp_path $HOME/disp_results/flyingthings3d/$1/disparities_pp.npy --gt_path /data/flyingthings3d/disparity/TEST/ --filenames_file /data/flyingthings3d/frames_cleanpass_webp/test_resized/flying_eval_a.txt --train_dataset $3
