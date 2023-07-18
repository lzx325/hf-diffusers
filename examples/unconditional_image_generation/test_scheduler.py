import torch
import numpy as np
import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, DDIMScheduler, UNet2DModel

def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    print(res.shape,broadcast_shape)
    return res.expand(broadcast_shape)

def main1():
    seed_all(10)
    noise_scheduler = DDIMScheduler()
    sample = torch.zeros(1, 1, 256, 256, dtype=torch.float32)
    noise = torch.randn(1, 1, 256, 256).type(torch.float32)
    bsz=10
    timesteps=torch.randint(
        0,100,(bsz,),dtype=torch.int64
    )
    res=_extract_into_tensor(noise_scheduler.alphas_cumprod,timesteps,(bsz,256,1,1))
    print(res.shape)

def main2():
    model = UNet2DModel(
        sample_size=64,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    print(model(torch.randn(1,3,64,64),timestep=torch.tensor([10],dtype=torch.int64))['sample'].shape)

def main3():
    
if __name__=="__main__":
    main2()

    