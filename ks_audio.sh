 export CUDA_VISIBLE_DEVICES=1
python main.py --ckpt_path ./results/ks/audio_TEST --modality audio --dataset  KineticSound --gpu_ids 0 --modulation Normal --alpha 0.8 --train