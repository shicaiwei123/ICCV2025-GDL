 export CUDA_VISIBLE_DEVICES=1

python main.py --ckpt_path ./results/ks/visual --modality visual --dataset  KineticSound --gpu_ids 1 --modulation Normal --alpha 0.8 --train --num_frame 3