
 export CUDA_VISIBLE_DEVICES=0
# python main.py --train --ckpt_path results/cramed/audio --alpha 0.1 --modulation Normal --pe 0 --gpu_ids 1 --modality audio  --gamma 0  --beta 0
python main.py --train --ckpt_path results/cramed/visual --alpha 0.1 --modulation Normal --pe 0 --gpu_ids 1 --modality visual --gamma 0  --beta 0 --learning_rate 0.001
# python main.py --train --ckpt_path results/cramed/full --alpha 0.1 --modulation Normal --pe 0 --gpu_ids 1 --modality full --gamma 0  --beta 0