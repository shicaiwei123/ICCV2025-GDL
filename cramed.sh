
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 1.5 --fusion_method sum 74.1
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 1.5 --fusion_method gated 73.4
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 1.5 --fusion_method film 63.5
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 2.0 71.7
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 2.5 75.1
#python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 0 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 3.0 73.0

python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 0  --beta 0 --gamma 0 --fusion_method film 57.5
python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 2.5 --fusion_method sum 72.1
python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 2.5 --fusion_method gated 75.7
python main.py --ckpt_path ./results/ks/full_normal --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 0.8 --train --num_frame 3 --pe 1  --beta 1e-5 --gamma 2.5 --fusion_method film 66.1

