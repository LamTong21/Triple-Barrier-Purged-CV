"""
src/diagnostics/spectral.py

Fast Fourier Transform (FFT) spectral density analysis for dominant market cycles.
"""

from typing import Dict
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq


class Layer4bSpectralCycleDiagnostics:
    """
    Identifies dominant cyclical frequencies and wavelengths in asset price trends.
    """

    @staticmethod
    def run(prices: pd.Series) -> Dict[str, int]:
        """
        Calculates dominant wavelength via linear detrending and FFT spectrum.

        Parameters
        ----------
        prices : pd.Series
            Adjusted price series.

        Returns
        -------
        Dict[str, int]
            Dominant cycle length in trading bars.
        """
        p = prices.dropna().values
        n = len(p)
        if n < 40:
            return {"dominant_cycle_len": 10}

        detrended = p - np.polyval(np.polyfit(np.arange(n), p, 1), np.arange(n))
        fft_vals = np.abs(rfft(detrended))
        freqs = rfftfreq(n, d=1.0)

        # Ignore DC component (zero frequency)
        fft_vals[0] = 0
        peak_idx = np.argmax(fft_vals)
        dominant_freq = freqs[peak_idx] if freqs[peak_idx] > 0 else 0.1
        dominant_cycle = int(np.clip(1.0 / dominant_freq, 3, 30))

        return {"dominant_cycle_len": dominant_cycle}