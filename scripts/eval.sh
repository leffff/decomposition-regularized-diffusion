## FFT:
ckpt_path=path

CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
    --checkpoint $ckpt_path \
    --download \
    --high 32 \
    --prior-type default \
    --num-samples 10_000 \
    --batch-size 512 \
    --model-type fft

CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
    --checkpoint $ckpt_path \
    --download \
    --high 16 \
    --prior-type default \
    --num-samples 10_000 \
    --batch-size 512 \
    --model-type fft

CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
    --checkpoint $ckpt_path \
    --download \
    --high 5 \
    --prior-type default \
    --num-samples 10_000 \
    --batch-size 512 \
    --model-type fft

## FM:
# ckpt_path=path

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm \
#     --fm-steps 5 \

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm \
#     --fm-steps 16 \

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm \
#     --fm-steps 32 \

# ## FM-x0, PAT-Loss:
# ckpt_path=path

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm_x0 \
#     --fm-steps 5 \

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm_x0 \
#     --fm-steps 16 \

# CUDA_VISIBLE_DEVICES=7 python fid_eval.py \
#     --checkpoint $ckpt_path \
#     --download \
#     --num-samples 10_000 \
#     --batch-size 512 \
#     --model-type fm_x0 \
#     --fm-steps 32 \