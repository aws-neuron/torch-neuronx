"""Torch-MLIR operation implementations organized by category."""

# Import all ops from submodules to maintain backward compatibility
from .activation import *  # noqa: F403
from .binary import *  # noqa: F403
from .comparison import *  # noqa: F403
from .convolution import *  # noqa: F403
from .copy import *  # noqa: F403
from .creation import *  # noqa: F403
from .foreach_ops import *  # noqa: F403
from .indexing import *  # noqa: F403
from .linear_algebra import *  # noqa: F403
from .logical import *  # noqa: F403
from .loss import *  # noqa: F403
from .misc import *  # noqa: F403
from .normalization import *  # noqa: F403
from .optimizer import *  # noqa: F403
from .reduction import *  # noqa: F403
from .tensor_ops import *  # noqa: F403
from .to_copy import *  # noqa: F403
from .unary import *  # noqa: F403
