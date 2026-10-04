"""
src/diagnostics/__init__.py

Data-Generating Process (DGP) diagnostics and econometric testing modules:
- Layer3DistributionalDiagnostics: Moments and Jarque-Bera normality tests.
- Layer4MemoryDependence: Hurst exponent (DFA) and Lo-MacKinlay Variance Ratio tests.
- Layer4bSpectralCycleDiagnostics: Fast Fourier Transform (FFT) dominant cycle analysis.
- Layer5FractionalIntegration: Optimal stationarity order d* via Fractional Differentiation.
- Layer6VolatilityDynamics & Layer6bVolatilityJumpDiagnostics: ARCH effect and BNS jump tests.
- Layer7cMicrostructureDiagnostics: Order book predictive power and spread stationarity.
- Layer8NonlinearDependence & Layer8bComplexityDiagnostics: BDS and Permutation Entropy.
- Stage2SingleAssetDGPScanner: Centralized execution scanner producing routing payloads.
"""

from src.diagnostics.distribution import Layer3DistributionalDiagnostics
from src.diagnostics.memory import Layer4MemoryDependence
from src.diagnostics.spectral import Layer4bSpectralCycleDiagnostics
from src.diagnostics.fractional import Layer5FractionalIntegration
from src.diagnostics.volatility import Layer6VolatilityDynamics, Layer6bVolatilityJumpDiagnostics
from src.diagnostics.microstructure import Layer7cMicrostructureDiagnostics
from src.diagnostics.nonlinearity import Layer8NonlinearDependence, Layer8bComplexityDiagnostics
from src.diagnostics.scanner import Stage2SingleAssetDGPScanner

__all__ = [
    "Layer3DistributionalDiagnostics",
    "Layer4MemoryDependence",
    "Layer4bSpectralCycleDiagnostics",
    "Layer5FractionalIntegration",
    "Layer6VolatilityDynamics",
    "Layer6bVolatilityJumpDiagnostics",
    "Layer7cMicrostructureDiagnostics",
    "Layer8NonlinearDependence",
    "Layer8bComplexityDiagnostics",
    "Stage2SingleAssetDGPScanner",
]