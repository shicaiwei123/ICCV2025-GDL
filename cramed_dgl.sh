
 export CUDA_VISIBLE_DEVICES=1
# python main_dgl.py --ckpt_path ./results/cramed/full_auxi_unimodal_grad_decouple_fusion --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 4 --train    --learning_rate 0.002 77.48
# python main_dgl.py --ckpt_path ./results/cramed/full_auxi_unimodal_grad_decouple_fusion --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 5 --train    --learning_rate 0.002 78.12



python main_dgl.py --ckpt_path ./results/cramed/full_auxi_unimodal_grad_decouple_fusion --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 4 --train    --learning_rate 0.002 
python main_dgl.py --ckpt_path ./results/cramed/full_auxi_unimodal_grad_decouple_fusion --modality full --dataset  CREMAD --gpu_ids 1 --modulation Normal --alpha 5 --train    --learning_rate 0.002 