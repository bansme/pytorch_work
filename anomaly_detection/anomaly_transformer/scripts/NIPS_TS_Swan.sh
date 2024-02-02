export CUDA_VISIBLE_DEVICES=1

python main.py --anormly_ratio 0.9 --num_epochs 3   --batch_size 128  --mode train --dataset NIPS_TS_Swan  --data_path dataset/NIPS_TS_Swan  --input_c 38
python main.py --anormly_ratio 0.9  --num_epochs 10     --batch_size 128   --mode test    --dataset NIPS_TS_Swan   --data_path dataset/NIPS_TS_Swan --input_c 38   --input_c 38    --pretrained_model 20
