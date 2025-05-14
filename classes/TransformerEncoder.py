from .TransformerEncoderBlock import TransformerEncoderBlock
from torch import nn


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])
