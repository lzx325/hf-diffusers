{
	source /home/liz0f/anaconda3/etc/profile.d/conda.sh
	conda deactivate
	conda activate diffusers

<<<<<<< HEAD
	out_dir="ddpm-ema-flowers-64-acc"
	rm -rf "$out_dir"
	option="$1"

	case "$option" in
	noslurm)
		n_gpus=1
		python -u -m accelerate.commands.launch \
			--num_processes=${n_gpus} \
			train_unconditional.py \
			--dataset_name="huggan/flowers-102-categories" \
			--resolution=64 \
			--center_crop --random_flip \
			--output_dir="${out_dir}" \
			--train_batch_size=32 \
			--num_epochs=50 \
			--gradient_accumulation_steps=1 \
			--use_ema \
			--learning_rate=1e-4 \
			--lr_warmup_steps=500 \
			--mixed_precision=no
	;;

	slurm)
		name="run-acc"
		time="1"
		n_gpus="8"
		memory="100"
		mkdir -p slurm/
		sbatch <<- EOF
			#!/bin/bash
			#SBATCH -N 1
			#SBATCH -J ${name}
			#SBATCH --partition=batch
			#SBATCH -o slurm/%J.out
			#SBATCH -e slurm/%J.err
			#SBATCH --time=${time}:00:00
			#SBATCH --mem=${memory}G
			#SBATCH --cpus-per-task=8
			#SBATCH --gres=gpu:${n_gpus}
			#run the application:

			python -u -m accelerate.commands.launch \
			--num_processes=${n_cpus} \
			train_unconditional.py \
			--dataset_name="huggan/flowers-102-categories" \
			--resolution=64 \
			--center_crop --random_flip \
			--output_dir="${out_dir}" \
			--train_batch_size=32 \
			--num_epochs=50 \
			--gradient_accumulation_steps=1 \
			--use_ema \
			--learning_rate=1e-4 \
			--lr_warmup_steps=500 \
			--mixed_precision=no

		EOF
	;;
	esac

}
=======
    out_dir="ddpm-ema-flowers-64-2gpu"
    rm -rf "$out_dir"

    # python -u -m accelerate.commands.launch \
    #     --num_processes=2 \
    #     train_unconditional.py \
    #     --dataset_name="huggan/flowers-102-categories" \
    #     --resolution=64 \
    #     --center_crop --random_flip \
    #     --output_dir="$out_dir" \
    #     --train_batch_size=32 \
    #     --checkpointing_steps=100 \
    #     --num_epochs=2 \
    #     --gradient_accumulation_steps=1 \
    #     --use_ema \
    #     --learning_rate=1e-4 \
    #     --lr_warmup_steps=500 \
    #     --mixed_precision=no
    
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 \
        --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=32 \
        --checkpointing_steps=100 \
        --num_epochs=2 \
        --gradient_accumulation_steps=1 \
        --use_ema \
        --learning_rate=1e-4 \
        --lr_warmup_steps=500 \
        --mixed_precision=no
}
>>>>>>> 3934cdcb7b63035f6f84003d7dfbb98e3fad5f90
