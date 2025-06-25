'''
Date: 2024-11-25 07:03:41
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2025-01-07 14:21:06
FilePath: /MineStudio/minestudio/models/groot_one/body.py
'''
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import numpy as np
from einops import rearrange, repeat
from typing import List, Dict, Any, Tuple, Optional, Union
import av

import timm
from huggingface_hub import PyTorchModelHubMixin

from minestudio.models.base_policy import MinePolicy
from minestudio.utils.vpt_lib.util import FanInInitReLULayer, ResidualRecurrentBlocks
from minestudio.utils.register import Registers

def peel_off(item: Union[List, str]) -> str:
    if isinstance(item, List):
        return peel_off(item[0])
    return item

class LatentSpace(nn.Module):

    def __init__(self, hiddim: int) -> None:
        super().__init__()
        self.encode_mu = nn.Linear(hiddim, hiddim)
        self.encode_log_var = nn.Linear(hiddim, hiddim)

    def sample(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.encode_mu(x)
        log_var = self.encode_log_var(x)
        if self.training:
            z = self.sample(mu, log_var)
        else:
            z = mu
        return { 'mu': mu, 'log_var': log_var, 'z': z }

class VideoEncoder(nn.Module):
    
    def __init__(
        self, 
        hiddim: int, 
        num_spatial_layers: int=2, 
        num_temporal_layers: int=2, 
        num_heads: int=8, 
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim*2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_spatial_layers, 
        )
        self.encode_video = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim*2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_temporal_layers
        )
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, images: torch.Tensor) -> Dict:
        """
        images: (b, t, c, h, w)
        """
        x = rearrange(images, 'b t c h w -> (b t) (h w) c')
        x = self.pooling(x)
        x = x.mean(dim=1) # (b t) c
        x = rearrange(x, '(b t) c -> b t c', b=images.shape[0])
        x = self.encode_video(x)
        x = x.mean(dim=1) # b c
        dist = self.encode_dist(x)
        return dist


class ImageEncoder(nn.Module):
    
    def __init__(self, hiddim: int, num_layers: int=2, num_heads: int=8, dropout: float=0.1) -> None:
        super().__init__()
        self.hiddim = hiddim
        self.pooling = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = hiddim,
                nhead = num_heads,
                dim_feedforward = hiddim*2,
                dropout = dropout,
                batch_first=True
            ),
            num_layers = num_layers, 
        )
        self.encode_dist = LatentSpace(hiddim)

    def forward(self, image: torch.Tensor) -> Dict:
        """
        image: (b, c, h, w)
        """
        x = rearrange(image, 'b c h w -> b (h w) c')
        x = self.pooling(x)
        x = x.mean(dim=1) # b c
        dist = self.encode_dist(x)
        return dist


class Decoder(nn.Module):
    
    def __init__(
        self, 
        hiddim: int, 
        num_heads: int = 8,
        num_layers: int = 4, 
        timesteps: int = 128, 
        mem_len: int = 128, 
    ) -> None:
        super().__init__()
        self.hiddim = hiddim
        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps, 
            recurrence_type="transformer", 
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4, 
            pointwise_use_activation=False, 
            attention_mask_style="clipped_causal", 
            attention_heads=num_heads,
            attention_memory_size=mem_len + timesteps,
            n_block=num_layers,
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear", batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

    def forward(self, x: torch.Tensor, memory: List) -> Tuple[torch.Tensor, List]:
        b, t = x.shape[:2]
        if not hasattr(self, 'first'):
            self.first = torch.tensor([[False]], device=x.device).repeat(b, t)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        x, memory = self.recurrent(x, self.first, memory)
        x = F.relu(x, inplace=False)
        x = self.lastlayer(x)
        x = self.final_ln(x)
        return x, memory

    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        if batch_size is None:
            return [t.squeeze(0).to(device) for t in self.recurrent.initial_state(1)]
        return [t.to(device) for t in self.recurrent.initial_state(batch_size)]

@Registers.model.register
class GrootPolicy(MinePolicy, PyTorchModelHubMixin):
    
    def __init__(
        self, 
        backbone: str='efficientnet_b0.ra_in1k', 
        freeze_backbone: bool=True,
        hiddim: int=1024,
        video_encoder_kwargs: Dict={}, 
        image_encoder_kwargs: Dict={},
        decoder_kwargs: Dict={},
        action_space=None,
    ):
        super().__init__(hiddim=hiddim, action_space=action_space)
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True)
        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda x: x / 255.0),
            torchvision.transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        ])
        num_features = self.backbone.feature_info[-1]['num_chs']
        self.updim = nn.Conv2d(num_features, hiddim, kernel_size=1)
        self.video_encoder = VideoEncoder(hiddim, **video_encoder_kwargs)
        self.image_encoder = ImageEncoder(hiddim, **image_encoder_kwargs)
        self.decoder = Decoder(hiddim, **decoder_kwargs)
        self.timesteps = decoder_kwargs['timesteps']
        self.fuser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim, 
                nhead=8, 
                dim_feedforward=hiddim*2, 
                dropout=0.1,
                batch_first=True
            ), 
            num_layers=2,
        )
        if freeze_backbone:
            print("Freezing backbone for GrootPolicy.")
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.condition_cache = {} # for infernce mode, to memory the generated conditions

    def encode_video(self, ref_video_path: str, resolution: Tuple[int, int] = (224, 224)) -> Dict:
        frames = []
        # ref_video_path = ref_video_path[0][0] # unbatchify

        with av.open(ref_video_path, "r") as container:
            for fid, frame in enumerate(container.decode(video=0)):
                frame = frame.reformat(width=resolution[0], height=resolution[1]).to_ndarray(format="rgb24")
                frames.append(frame)

        reference = torch.from_numpy(np.stack(frames[:self.timesteps], axis=0) ).unsqueeze(0).to(self.device)
        reference = rearrange(reference, 'b t h w c -> (b t) c h w')
        reference = self.transforms(reference)
        reference = self.backbone(reference)[-1]
        reference = self.updim(reference)
        reference = rearrange(reference, '(b t) c h w -> b t c h w', b=1)
        posterior_dist = self.video_encoder(reference)
        prior_dist = self.image_encoder(reference[:, 0])

        posterior_dist['z'] = posterior_dist['z'].unsqueeze(0)
        prior_dist['z'] = prior_dist['z'].unsqueeze(0)

        print(
            "=======================================================\n"
            f"Ref video is from: {ref_video_path};\n"
            f"Num frames: {len(frames)}. \n"
            "=======================================================\n"
        )

        print(f"[📚] latent shape: {posterior_dist['z'].shape} | mean: {posterior_dist['z'].mean().item(): .3f} | std: {posterior_dist['z'].std(): .3f}")

        condition = {
            "posterior_dist": posterior_dist,
            "prior_dist": prior_dist
        }

        return condition

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:
        b, t = input['image'].shape[:2]

        image = rearrange(input['image'], 'b t h w c -> (b t) c h w')
        image = self.transforms(image)
        image = self.backbone(image)[-1]
        image = self.updim(image)
        image = rearrange(image, '(b t) c h w -> b t c h w', b=b)

        if 'ref_video_path' in input:
            # input has `ref_video_path`, means inference mode
            ref_video_path = peel_off(input['ref_video_path'])
            if ref_video_path not in self.condition_cache:
                self.condition_cache[ref_video_path] = self.encode_video(ref_video_path)
            condition = self.condition_cache[ref_video_path]
            posterior_dist = condition['posterior_dist']
            prior_dist = condition['prior_dist']
            z = posterior_dist['z']
        else:
            # self-supervised training
            reference = image
            posterior_dist = self.video_encoder(reference)
            prior_dist = self.image_encoder(reference[:, 0])
            z = repeat(posterior_dist['z'], 'b c -> (b t) 1 c', t=t)

        x = rearrange(image, 'b t c h w -> (b t) (h w) c')
        x = torch.cat([x, z], dim=1)
        x = self.fuser(x)
        x = x.mean(dim=1) # (b t) c
        x = rearrange(x, '(b t) c -> b t c', b=b)
        x, memory = self.decoder(x, memory)
        pi_h = v_h = x
        pi_logits = self.pi_head(pi_h)
        vpred = self.value_head(v_h)
        latents = {
            "pi_logits": pi_logits, 
            "vpred": vpred, 
            "posterior_dist": posterior_dist, 
            "prior_dist": prior_dist
        }
        return latents, memory

    def initial_state(self, *args, **kwargs) -> Any:
        return self.decoder.initial_state(*args, **kwargs)

@Registers.model_loader.register
def load_groot_policy(ckpt_path: str = None):
    if ckpt_path is None:
        repo_id = "CraftJarvis/MineStudio_GROOT.18w_EMA"
        return GrootPolicy.from_pretrained("CraftJarvis/MineStudio_GROOT.18w_EMA")

    ckpt = torch.load(ckpt_path)
    model = GrootPolicy(**ckpt['hyper_parameters']['model'])
    state_dict = {k.replace('mine_policy.', ''): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model

if __name__ == '__main__':
    model = load_groot_policy()
    model = GrootPolicy(
        backbone='timm/vit_base_patch32_clip_224.openai', 
        hiddim=1024,
        freeze_backbone=False,
        video_encoder_kwargs=dict(
            num_spatial_layers=2,
            num_temporal_layers=4,
            num_heads=8,
            dropout=0.1
        ),
        image_encoder_kwargs=dict(
            num_layers=2,
            num_heads=8,
            dropout=0.1
        ),
        decoder_kwargs=dict(
            num_layers=4,
            timesteps=128,
            mem_len=128
        )
    ).to("cuda")
    memory = None
    input = {
        'image': torch.zeros((2, 128, 224, 224, 3), dtype=torch.uint8).to("cuda"),
    }
    output, memory = model(input, memory)