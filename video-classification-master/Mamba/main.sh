source /home/yjzhang/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1

# The first step: train video layers
python main.py --device cuda:6 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 100
#python main_key_frame_selection.py --device cuda:4 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 100  --epoch 60
#python main_distillation.py --device cuda:4 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 50

# The second step: train image layers
#python main.py --device cuda:4 --root /nfs/yajie/video_fujian --batch_size 64 --n_frame 50  --epoch 10

## The second step: train video layers
#python main.py --device cuda:5 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 10
#OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes=1 --nproc_per_node=2 --master_port 29500 main.py --device cuda --root /nfs/yajie/video_fujian --batch_size 4 --n_frame 50  --epoch 50

# Training for feature selection
#python main_feature_selection.py --device cuda:5 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 50
#python main_feature.py --device cuda:5 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 10

#python test.py --device cuda:5 --root /nfs/yajie/video_fujian --batch_size 8 --n_frame 50  --epoch 10