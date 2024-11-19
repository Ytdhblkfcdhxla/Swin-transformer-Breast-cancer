# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SWINNet(nn.Module):
    def __init__(self, dim=768):
        super(SWINNet, self).__init__()
        patch_size = ensure_tuple_rep(4, 3)
        window_size = ensure_tuple_rep(7, 3)
        self.swinViT = SwinViT(
            in_chans=1,
            embed_dim=48,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 18, 2],
            # num_heads=[3, 6, 12, 24],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=False,
            spatial_dims=3,
        )

        self.norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.head = nn.Linear(dim, 3)

        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

        nn.init.constant_(self.norm.bias, 0)
        nn.init.constant_(self.norm.weight, 1.0)

    def forward(self, x):
        x_out = self.swinViT(x)
        x_out = x_out[-1]
        x = self.avgpool(x_out)
        x = x.flatten(start_dim=2, end_dim=4).squeeze(-1)
        x = self.norm(x)  # B L C
        x = self.head(x)
        return x
