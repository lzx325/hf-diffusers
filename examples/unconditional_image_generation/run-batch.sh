{
    source /home/liz0f/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    conda activate diffusers

    out_dir="ddpm-ema-flowers-64-default"
    rm -r "$out_dir"
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

    out_dir="ddpm-ema-flowers-64-noema"
    rm -r "$out_dir"
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=32 \
        --num_epochs=50 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-4 \
        --lr_warmup_steps=500 \
        --mixed_precision=no

    out_dir="ddpm-ema-flowers-64-lr=1e-5"
    rm -r "$out_dir"
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=32 \
        --num_epochs=50 \
        --gradient_accumulation_steps=1 \
        --use_ema \
        --learning_rate=1e-5 \
        --lr_warmup_steps=500 \
        --mixed_precision=no
    
    out_dir="ddpm-ema-flowers-64-lr=1e-3"
    rm -r "$out_dir"
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=32 \
        --num_epochs=50 \
        --gradient_accumulation_steps=1 \
        --use_ema \
        --learning_rate=1e-3 \
        --lr_warmup_steps=500 \
        --mixed_precision=no

    out_dir="ddpm-ema-flowers-64-lr_scheduler=constant"
    rm -r "$out_dir"
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=32 \
        --num_epochs=50 \
        --gradient_accumulation_steps=1 \
        --use_ema \
        --learning_rate=1e-4 \
        --lr_scheduler=constant \
        --mixed_precision=no

    out_dir="ddpm-ema-flowers-64-mixed_precision=yes"
    rm -r "$out_dir"
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
        --mixed_precision=yes

    out_dir="ddpm-ema-flowers-64-train_batch_size=128"
    rm -r "$out_dir"
    python train_unconditional.py \
        --dataset_name="huggan/flowers-102-categories" \
        --resolution=64 --center_crop --random_flip \
        --output_dir="$out_dir" \
        --train_batch_size=128 \
        --num_epochs=50 \
        --gradient_accumulation_steps=1 \
        --use_ema \
        --learning_rate=1e-4 \
        --lr_warmup_steps=500 \
        --mixed_precision=no
}