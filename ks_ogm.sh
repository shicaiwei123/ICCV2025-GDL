
# export CUDA_VISIBLE_DEVICES=0
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 1 --modulation OGM_GE --alpha 0.8 --train --num_frame 3


python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 1.0 72.2
python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 2.0 74.4
python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --drop 0 --gamma 2.5 74.0
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-6 --drop 0 --gamma 2.0
