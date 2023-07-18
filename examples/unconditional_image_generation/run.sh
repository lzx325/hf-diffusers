{
    set -e 
    source /home/liz0f/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate diffusers
    mode="$1"
    case "$mode" in
        "train")
            out_dir="ddpm-ema-flowers-64-debug"
            if [ -d "$out_dir" ]; then
                rm -r "$out_dir"
            fi
            python train_unconditional.py \
                --dataset_name="huggan/flowers-102-categories" \
                --resolution=64 --center_crop --random_flip \
                --output_dir="$out_dir" \
                --train_batch_size=32 \
                --num_epochs=50 \
                --gradient_accumulation_steps=1 \
                --use_ema \
                --learning_rate=1e-4 \
                --lr_warmup_steps=500 \
                --mixed_precision=no
                ;;
        "inference")
            python run_inference.py 
            ;;
        "debug")
            python test_scheduler.py
            ;;
        *)
            echo "Unknown mode: $mode"
            exit 1
            ;;
    esac
}