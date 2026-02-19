# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle
import paddle.nn as nn

from ppdet.core.workspace import register, serializable


def _fmod_2pi(z):
    """Compute z mod 2π, result in [0, 2π)."""
    period = 2 * math.pi
    return z - paddle.floor(z / period) * period


def triangular_wave(z):
    """
    Triangular wave function, periodic with period 2π, range [-1, 1].

    Definition (extended periodically):
      tri(z) =  2z/π              for z in [0, π/2)
               -2z/π + 2          for z in [π/2, 3π/2)
                2z/π - 4          for z in [3π/2, 2π)
    """
    z_mod = _fmod_2pi(z)
    pi = math.pi
    return paddle.where(
        z_mod < 0.5 * pi,
        (2.0 * z_mod) / pi,
        paddle.where(
            z_mod < 1.5 * pi,
            (-2.0 * z_mod) / pi + 2.0,
            (2.0 * z_mod) / pi - 4.0
        )
    )


def square_wave(z):
    """
    Square wave function: -1 in [0, π), +1 in [π, 2π).
    Periodic with period 2π.
    """
    z_mod = _fmod_2pi(z)
    return paddle.where(
        z_mod < math.pi,
        paddle.full_like(z, -1.0),
        paddle.full_like(z, 1.0)
    )


def sawtooth_wave(z):
    """
    Sawtooth wave function, periodic with period 2π.
    Linear ramp: z in [0, π], z - 2π in (π, 2π).
    """
    z_mod = _fmod_2pi(z)
    return paddle.where(z_mod <= math.pi, z_mod, z_mod - 2 * math.pi)


def get_periodic_funcs(periodic_func='sinusoid'):
    """
    Returns a pair of functions (func_even, func_odd) for the given
    periodic function type.

    For 'sinusoid': sin and cos (90° phase shift).
    For alternatives: the same function is used for both, consistent
    with the approach in alt-positional-encoding-transformer.

    Args:
        periodic_func (str): One of 'sinusoid', 'triangular', 'square',
                             'sawtooth'.

    Returns:
        tuple: (func_even, func_odd)
    """
    if periodic_func == 'sinusoid':
        return paddle.sin, paddle.cos
    elif periodic_func == 'triangular':
        return triangular_wave, triangular_wave
    elif periodic_func == 'square':
        return square_wave, square_wave
    elif periodic_func == 'sawtooth':
        return sawtooth_wave, sawtooth_wave
    else:
        raise ValueError(
            f"Unknown periodic function: {periodic_func}. "
            f"Supported: 'sinusoid', 'triangular', 'square', 'sawtooth'."
        )


@register
@serializable
class PositionEmbedding(nn.Layer):
    def __init__(self,
                 num_pos_feats=128,
                 temperature=10000,
                 normalize=True,
                 scale=2 * math.pi,
                 embed_type='sine',
                 num_embeddings=50,
                 offset=0.,
                 eps=1e-6,
                 periodic_func='sinusoid'):
        super(PositionEmbedding, self).__init__()
        assert embed_type in ['sine', 'learned']

        self.embed_type = embed_type
        self.offset = offset
        self.eps = eps
        self.periodic_func = periodic_func
        if self.embed_type == 'sine':
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            self.scale = scale
        elif self.embed_type == 'learned':
            self.row_embed = nn.Embedding(num_embeddings, num_pos_feats)
            self.col_embed = nn.Embedding(num_embeddings, num_pos_feats)
        else:
            raise ValueError(f"{self.embed_type} is not supported.")

    def forward(self, mask):
        """
        Args:
            mask (Tensor): [B, H, W]
        Returns:
            pos (Tensor): [B, H, W, C]
        """
        if self.embed_type == 'sine':
            y_embed = mask.cumsum(1)
            x_embed = mask.cumsum(2)
            if self.normalize:
                y_embed = (y_embed + self.offset) / (
                    y_embed[:, -1:, :] + self.eps) * self.scale
                x_embed = (x_embed + self.offset) / (
                    x_embed[:, :, -1:] + self.eps) * self.scale

            dim_t = 2 * (paddle.arange(self.num_pos_feats) //
                         2).astype('float32')
            dim_t = self.temperature**(dim_t / self.num_pos_feats)

            pos_x = x_embed.unsqueeze(-1) / dim_t
            pos_y = y_embed.unsqueeze(-1) / dim_t
            func_even, func_odd = get_periodic_funcs(self.periodic_func)
            pos_x = paddle.stack(
                (func_even(pos_x[:, :, :, 0::2]), func_odd(pos_x[:, :, :, 1::2])),
                axis=4).flatten(3)
            pos_y = paddle.stack(
                (func_even(pos_y[:, :, :, 0::2]), func_odd(pos_y[:, :, :, 1::2])),
                axis=4).flatten(3)
            return paddle.concat((pos_y, pos_x), axis=3)
        elif self.embed_type == 'learned':
            h, w = mask.shape[-2:]
            i = paddle.arange(w)
            j = paddle.arange(h)
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            return paddle.concat(
                [
                    x_emb.unsqueeze(0).tile([h, 1, 1]),
                    y_emb.unsqueeze(1).tile([1, w, 1]),
                ],
                axis=-1).unsqueeze(0)
        else:
            raise ValueError(f"not supported {self.embed_type}")
