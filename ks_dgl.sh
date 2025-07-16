
 export CUDA_VISIBLE_DEVICES=0



# python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 2 --train   --num_frame 3  --learning_rate 0.002 74.78
# python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 3 --train   --num_frame 3  --learning_rate 0.002 75.10
# python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 3 --train   --num_frame 3  --learning_rate 0.002 76.28


python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 2 --train   --num_frame 3  --learning_rate 0.002 
python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 3 --train   --num_frame 3  --learning_rate 0.002 
python main_dgl.py --ckpt_path ./results/ks/full_normal --modality full --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 3 --train   --num_frame 3  --learning_rate 0.002 