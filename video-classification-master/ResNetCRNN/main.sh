source /home/yjzhang/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1

python main.py --device cuda:7 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50 --epoch 100