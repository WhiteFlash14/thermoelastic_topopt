"""thermotopo - thermo-elastic compliance topology optimization (Meng et al. 2021, Eq. 23)
for arbitrary 2D structures described in a YAML/JSON problem file."""
from .config import build, load_file, apply_overrides  # noqa: F401
from .optimizer import optimize, OptimizerSettings  # noqa: F401
from .problem import ThermoelasticCompliance  # noqa: F401

__version__ = '1.0.0'
