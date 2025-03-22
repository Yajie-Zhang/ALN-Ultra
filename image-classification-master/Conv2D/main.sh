source /home/yjzhang/anaconda3/etc/profile.d/conda.sh
conda activate pytorch1

python main.py --device cuda:5 --root yourpath --batch_size 8 --n_frame 100 --lr 0.0001 --epoch 100
