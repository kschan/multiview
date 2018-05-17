mkdir -p $HOME/disp_results/rover

filenames_file='/data/rover_dataset/highbay_1.txt'

python monodepth_main.py --encoder $2 --mode test --data_path /data/rover_dataset/ --filenames_file $filenames_file --log_directory $HOME/logs/ --model_name $1 --output_directory $HOME/disp_results/rover/ --num_gpus 1  
#python utils/evaluate_rover.py --predicted_disp_path $HOME/disp_results/rover/$1/disparities_pp.npy --data_path /data/rover_dataset/ --filenames_file $filenames_file
