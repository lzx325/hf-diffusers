import os
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
import datasets
from torchvision import transforms

import accelerate
import diffusers
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline, DDIMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

if __name__=="__main__":
    args=argparse.Namespace()
    args.ddpm_num_steps=1000
    args.ddpm_beta_schedule="linear"
    args.prediction_type="epsilon"
    args.eval_batch_size=4
    args.ddpm_num_inference_steps=80

    ckpt_dir="ddpm-ema-flowers-64/checkpoint-3000"
    unet = UNet2DModel.from_pretrained(ckpt_dir,subfolder="unet")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )
    
    pipeline = DDIMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    )

    generator = torch.Generator(device=pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    images = pipeline(
        generator=generator,
        batch_size=args.eval_batch_size,
        num_inference_steps=args.ddpm_num_inference_steps,
        output_type="pil",
    ).images
    
    # save the images in images to a folder 
    os.makedirs(f"{ckpt_dir}/samples", exist_ok=True)
    for i, image in enumerate(images):
        image.save(f"{ckpt_dir}/samples/sample-{i}.png")
