from .task_base import Task

from .mbpp.task_mbpp import (
    MBPP_Task
)

from .humaneval.task_humaneval import (
    HumanEval_Task
)

from .gsmhard.task_gsmhard import (
    GSMHARD_Task
)

from .mgsm.task_mgsm import (
    MGSM_Task
)

__all__ = ['Task', 'MBPP_Task', 'HumanEval_Task', 'GSMHARD_Task', 'MGSM_Task']