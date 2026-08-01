"""masking — MLM-style corruption of token batches.

Every masker is one composition of three orthogonal strategies: a rate policy
(how much to select), a selector (which positions), and a corruption scheme
(what to do with them). `Corruptor` is the orchestrator; `presets` holds the
familiar named combinations.

The backend import must stay first: the modules below reach back for
``takumi.masking.backend`` while this package is still initializing, so the
name has to be bound before they are imported.
"""

from takumi.masking.backend import backend

# Composable corruptor + strategies
from takumi.masking.context import CallContext
from takumi.masking.corruptor import Corruptor
from takumi.masking.corruptor import MaskingResult
from takumi.masking.rates import RatePolicy
from takumi.masking.rates import FixedRate
from takumi.masking.rates import BetaRate
from takumi.masking.rates import SampledRate
from takumi.masking.rates import ScheduleRate
from takumi.masking.selectors import Selector
from takumi.masking.selectors import Bernoulli
from takumi.masking.selectors import FocusBiased
from takumi.masking.selectors import Spans
from takumi.masking.selectors import coerce_focus
from takumi.masking.selectors import focus_from_reference
from takumi.masking.schemes import CorruptionScheme
from takumi.masking.schemes import Mixed
from takumi.masking.schemes import MaskOnly
from takumi.masking.schemes import SplitMasks
from takumi.masking.schedules import MaskSchedule
from takumi.masking.schedules import LinearSchedule
from takumi.masking.schedules import CosineSchedule
from takumi.masking.schedules import SqrtSchedule

# Named preset compositions
from takumi.masking.presets import FixedMasker
from takumi.masking.presets import BetaMasker
from takumi.masking.presets import CategoricalMasker
from takumi.masking.presets import DiffusionMasker
from takumi.masking.presets import HybridSpanMasker

# Curriculum sampling + schedulers that feed SampledRate
from takumi.masking.sampler import MLMProbabilitySampler
from takumi.masking.scheduler import EMAScheduler
from takumi.masking.scheduler import LinearScheduler
from takumi.masking.scheduler import Scheduler
from takumi.masking.scheduler import StagedLinearScheduler
from takumi.masking.utils import create_default_mlm_weights
from takumi.masking.utils import ensure_non_negative
from takumi.masking.utils import seed_from_components

__all__ = [
    "backend",
    "CallContext",
    "Corruptor",
    "MaskingResult",
    "RatePolicy",
    "FixedRate",
    "BetaRate",
    "SampledRate",
    "ScheduleRate",
    "Selector",
    "Bernoulli",
    "FocusBiased",
    "Spans",
    "coerce_focus",
    "focus_from_reference",
    "CorruptionScheme",
    "Mixed",
    "MaskOnly",
    "SplitMasks",
    "MaskSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "SqrtSchedule",
    "FixedMasker",
    "BetaMasker",
    "CategoricalMasker",
    "DiffusionMasker",
    "HybridSpanMasker",
    "MLMProbabilitySampler",
    "EMAScheduler",
    "LinearScheduler",
    "Scheduler",
    "StagedLinearScheduler",
    "create_default_mlm_weights",
    "ensure_non_negative",
    "seed_from_components",
]
