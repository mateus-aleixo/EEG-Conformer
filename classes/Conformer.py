from .ClassificationHead import ClassificationHead
from .PatchEmbedding import PatchEmbedding
from .TransformerEncoder import TransformerEncoder
from torch import nn


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes),
        )
