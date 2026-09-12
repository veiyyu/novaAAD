"""Auditory attention decoding and bounded audio control."""

from .config import AuditoryConfig
from .controller import AttentionController
from .decoder import RidgeDecoder

__all__ = ["AttentionController", "AuditoryConfig", "RidgeDecoder"]
