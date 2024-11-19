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
from .swin_transformer_2d import SwinTransformer_Encoder


# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SWINNet_Double_2d(nn.Module):
    def __init__(self, dim=768, num_classes=3):
        super(SWINNet_Double_2d, self).__init__()

        self.swinViT1 = SwinTransformer_Encoder(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=1,
            num_classes=num_classes,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )

        self.swinViT2 = SwinTransformer_Encoder(
            hidden_dim=96,
            layers=(2, 2, 6, 2),
            heads=(3, 6, 12, 24),
            channels=1,
            num_classes=num_classes,
            head_dim=32,
            window_size=7,
            downscaling_factors=(4, 2, 2, 2),
            relative_pos_embedding=True
        )

        self.head = nn.Linear(dim * 2, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=.02)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x1, x2):
        x_out1 = self.swinViT1(x1)
        x_out2 = self.swinViT2(x2)

        x = torch.cat([x_out1, x_out2], dim=1)
        x = self.head(x)
        return x


if __name__ == '__main__':
    model = SWINNet_Double_2d()
    input = torch.randn(size=(1, 1, 224, 224))
    x = model(input, input)
    print(x.shape)
