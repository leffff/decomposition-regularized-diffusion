conda activate ~/chikovani/envs/kandinsky-cuda12.8/

CUDA_VISIBLE_DEVICES=6 python fft_cifar_train.py \
    --experiment-name fft_cifar \
    --num-epochs 1000 \
    --log-dir logs_cifar