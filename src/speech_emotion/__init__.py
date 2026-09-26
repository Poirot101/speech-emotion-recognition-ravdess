"""Speech emotion recognition on RAVDESS.

MFCC/chroma/mel features averaged over time, classified with a scikit-learn MLP.
"""

from .features import extract_feature, feature_dimension
from .dataset import (
    EMOTIONS,
    OBSERVED_EMOTIONS,
    load_dataset,
    parse_ravdess_filename,
)

__version__ = "1.0.0"

__all__ = [
    "extract_feature",
    "feature_dimension",
    "load_dataset",
    "parse_ravdess_filename",
    "EMOTIONS",
    "OBSERVED_EMOTIONS",
]
