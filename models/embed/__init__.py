"""
Embedding modules for VirtualRecursiveNetwork.

- Sparse embeddings for puzzle identifiers
- Patch embeddings for vision-based models
"""

from models.embed.sparse import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from models.embed.patch import (
    StandardPatchEmbed,
    MetricPatchEmbed,
    create_patch_embed,
)

__all__ = [
    "CastedSparseEmbedding",
    "CastedSparseEmbeddingSignSGD_Distributed",
    "StandardPatchEmbed",
    "MetricPatchEmbed",
    "create_patch_embed",
]
