# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""SARNet self-developed modules.

SARNet (Structure-Aware Refinement Network) extends Ultralytics YOLO26 with:

- ``LoGStem``: a Laplacian-of-Gaussian edge-enhanced stem (``stem/LoG.py``).
- ``SDS``: Semantic-guided Dual-path Structure downsampling (``my_module/sds.py``).
"""

from .my_module.sds import SDS, SemanticGate, StructureBranch
from .stem.LoG import Gaussian, LoGStem

__all__ = ("LoGStem", "SDS", "Gaussian", "SemanticGate", "StructureBranch")
