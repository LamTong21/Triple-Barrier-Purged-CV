"""
src/data_pipeline/__init__.py

Data ingestion, geometric audit, microstructure topologies, and labeling module:
- Layer1DataIntegrity: Point-in-time synthetic total return and geometric validation.
- Layer2MicrostructureTopologies: Multi-level order book depth and micro-price proxies.
- Layer0TripleBarrier: Causal EWMA volatility and dynamic horizontal/vertical barriers.
- MultiAssetDataPreparer: Multi-asset OHLCV fetcher and synthetic portfolio aggregator.
"""

from src.data_pipeline.audit import Layer1DataIntegrity
from src.data_pipeline.topology import Layer2MicrostructureTopologies
from src.data_pipeline.labeling import Layer0TripleBarrier
from src.data_pipeline.loader import MultiAssetDataPreparer

__all__ = [
    "Layer1DataIntegrity",
    "Layer2MicrostructureTopologies",
    "Layer0TripleBarrier",
    "MultiAssetDataPreparer",
]