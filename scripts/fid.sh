
# /home/jovyan/.mlspace/envs/avarlamov-bagel

# for step in {1..5}; do
#     python fid.py --path1 MNIST/generate_${step} --path2 MNIST/fid_ref --device cuda --output_csv fid_results.csv
# done


# for step in {1..31}; do
#     python fid.py --path1 MNIST/fm_generate_${step} --path2 MNIST/fid_ref --device cuda --output_csv fm_fid_results.csv
# done

# for step in {1..31}; do
#     python fid.py --path1 MNIST/ddim_generate_${step} --path2 MNIST/fid_ref --device cuda --output_csv ddim_fid_results.csv
# done

#################
conda activate ~/chikovani/envs/kandinsky-cuda12.8

root=/home/jovyan/novitskiy/Research/decomposition-regularized-diffusion/MNIST
folders=(
    # generate_fm_x0_energy_aligned_svd
    # generate_fm_velocity
    # generate_fm_x0
    # generate_fm_x0_lin_aligned_svd
    # generate_fm_x0_random_svd
    # generate_fm_x0_lin_aligned_fft

    # generate_fm_x0_sqrt_aligned_svd
    # generate_fm_x0_poly_aligned_svd
    # generate_fm_x0_lin_aligned_svd_v2
)

for folder in ${folders[@]}; do 
    for step in {1..31}; do
        python fid.py --path1 ${root}/${folder}/${step} --path2 MNIST/fid_ref --device cuda --output_csv ${folder}_fid_results.csv
    done
done
