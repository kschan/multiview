mkdir -p $HOME/disp_results/umich

filenames_file='/data/umich/2012-02-12/filenames_short.txt'

python monodepth_main.py --encoder $2 --mode test --data_path /data/umich/2012-02-12/ --filenames_file $filenames_file --log_directory $HOME/logs/ --model_name $1 --output_directory $HOME/disp_results/umich/ --num_gpus 1  
python utils/evaluate_umich.py --predicted_disp_path $HOME/disp_results/umich/$1/disparities_pp.npy --data_path /data/umich/2012-02-12/velodyne_np/ --filenames_file $filenames_file
