"""Endogenous Credibility in Social Learning -- reference implementation.

A community-structured model of repeated social learning in which confidence
is generated endogenously by a drift--diffusion decision process and then acts
as a credibility weight in two social channels (anticipatory and
retrospective).  See the accompanying manuscript and README.
"""
from . import ddm, confidence, network, config, model, metrics, uncertainty, scenarios, meanfield
from .config import Params, base_params, SCALE

__all__ = ["ddm", "confidence", "network", "config", "model", "metrics",
           "uncertainty", "scenarios", "meanfield", "Params", "base_params", "SCALE"]
__version__ = "1.0.0"
