"""Load the pinned author model and return normalized style codes for actual images."""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from style_lab import PROVENANCE
from vendor.aladin.vit_networks import VisionTransformer


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


class StyleEncoder(torch.nn.Module):
    def __init__(self, root, device):
        super().__init__()
        self.device=device
        root=Path(root)
        weights=root/'models/style-lab/aladinVIT_bamfg_64.47.pt'
        code=root/'vendor/aladin/vit_networks.py'
        manifest=PROVENANCE['style']
        if digest(weights)!=manifest['weights_sha256'] or digest(code)!=manifest['code_sha256']:
            raise ValueError('ALADIN code or checkpoint mismatch')
        self.enc_style=VisionTransformer()
        try:
            state=torch.load(weights,map_location='cpu',weights_only=True)
        except TypeError:  # Older runtime, same verified official checkpoint.
            state=torch.load(weights,map_location='cpu')
        self.load_state_dict(state,strict=True)
        self.eval().to(device)
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),
                                           transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

    @torch.inference_mode()
    def encode(self, images):
        batch=[]
        for image in images:
            if image.width*image.height>16_000_000:
                raise ValueError('Image too large')
            rgba=ImageOps.exif_transpose(image).convert('RGBA')
            white=Image.new('RGBA',rgba.size,'white')
            white.alpha_composite(rgba)
            batch.append(self.transform(white.convert('RGB')))
        code=self.enc_style(torch.stack(batch).to(self.device),just_style_code=True)[0]
        vec=torch.nn.functional.normalize(code,dim=-1).cpu().numpy()
        if vec.shape!=(len(images),256) or not np.isfinite(vec).all() or np.any(np.linalg.norm(vec,axis=1)<.99):
            raise ValueError('Invalid ALADIN result')
        return vec
