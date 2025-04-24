import functools
import math
from typing import Callable, List, Union, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from ocl import utils


@utils.make_build_fn(__name__, "scheduler")
def build(
    config,
    name: str,
):
    if name == "constant":
        scheduler_fn = constant
    elif name == "linear_warmup":
        scheduler_fn = functools.partial(
            linear_warmup, warmup_steps=config.get("warmup_steps", 1000)
        )
    if name == "exp_decay_with_warmup":
        scheduler_fn = functools.partial(
            exp_decay_with_warmup,
            warmup_steps=config.get("warmup_steps", 1000),
            decay_steps=config.get("decay_steps", 100000),
            decay_rate=config.get("decay_rate", 0.5),
        )
    elif name == "exp_decay_with_warmup_and_restarts":
        scheduler_fn = functools.partial(
            exp_decay_with_warmup,
            warmup_steps=config.get("warmup_steps", 1000),
            cycle_steps=config.get("cycle_steps", 100000),
            decay_rate=config.get("decay_rate", 0.5),
            restarts_factor=config.get("restarts_factor", 2.0),
            decay_base_scale=config.get("decay_base_scale", 0.9)
        )
    elif name == "cosine_decay_with_warmup":
        scheduler_fn = functools.partial(
            cosine_decay_with_warmup,
            warmup_steps=config.get("warmup_steps", 1000),
            decay_steps=config.get("decay_steps", 100000),
            min_scale=config.get("min_scale", None)
        )
    elif name == "cosine_annealing_with_warmup_and_restarts":
        scheduler_fn = functools.partial(
            cosine_annealing_with_warmup_and_restarts,
            warmup_steps=config.get("warmup_steps", 1000),
            cycle_steps=config.get("cycle_steps", 100000),
            restarts_factor=config.get("restarts_factor", 1.0),
            decay_rate=config.get("decay_rate", 1.0),
            bench=config.get("bench", 10000),
            min_scale=config.get("min_scale", None)
        )
    elif name == "cosine_decay_with_exponential_tail":
        scheduler_fn = functools.partial(
            cosine_decay_with_exponential_tail,
            warmup_steps=config.get("warmup_steps", 5000),
            decay_steps=config.get("decay_steps", 50000),
            min_scale=config.get("min_scale", 0.2),
            exp_decay_rate=config.get("exp_decay_rate", 0.65),
            transition_start_ratio=config.get("transition_start_ratio", 0.9),
            transition_length_ratio=config.get("transition_length_ratio", 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler {name}")

    return scheduler_fn


def apply_schedule_fn_to_optimizer(
    optimizer: torch.optim.Optimizer,
    decay_fn: Union[Callable[[int], float], List[Callable[[int], float]]],
) -> LambdaLR:
    return LambdaLR(optimizer, decay_fn)


def constant(step: int) -> float:
    """Constant schedule.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    return 1.0


def linear_warmup(step: int, warmup_steps: int) -> float:
    """Linear warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if warmup_steps > 0:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0


def exp_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    decay_rate: float,
) -> float:
    """Exponential decay with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules. After `decay_steps`,
    factor equals `decay_rate`.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        decay_steps = decay_steps - warmup_steps
        return decay_rate ** (step / decay_steps)


def exp_decay_with_warmup_and_restarts(
    step: int,
    warmup_steps: int,
    cycle_steps: int,
    restarts_factor: float = 2.0,
    decay_rate: float = 0.5,
    decay_base_scale: float = 0.8
) -> float:
    """Exponential decay with linear warmup and restarts at 1.0.
    """

    if step <= warmup_steps + cycle_steps:
        if step < warmup_steps:
            return linear_warmup(step, warmup_steps)
        else:
            step_in_cycle = step - warmup_steps
            
            return decay_rate**(step_in_cycle / cycle_steps)
    else:
        first_cycle_length = cycle_steps + warmup_steps
        cycle_length = int(first_cycle_length * restarts_factor)
        step_in_cycle = step - first_cycle_length - ((step - first_cycle_length) // cycle_length) * cycle_length
            
        return decay_base_scale**((step - first_cycle_length) // cycle_length + 1) * decay_rate**(step_in_cycle / cycle_length)


def cosine_decay_with_warmup(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    min_scale: float,
) -> float:
    """Cosine decay to zero or to min lr with linear learning rate warmup.

    Function maps current step or epoch to factor of learning rate schedules.
    """
    if step < warmup_steps:
        return linear_warmup(step, warmup_steps)
    else:
        step = step - warmup_steps
        decay_steps = decay_steps - warmup_steps
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (step / decay_steps)))

        if min_scale is None:
            return 1.0 * cosine_decay
        else:
            return min_scale + (1.0 - min_scale) * cosine_decay


def cosine_annealing_with_warmup_and_restarts(
    step: int,
    warmup_steps: int,
    cycle_steps: int,
    restarts_factor: float = 2.0,
    decay_rate: float = 1.0,
    bench: int = 25000,
    min_scale: Optional[float] = None
) -> float:
    """Cosine annealing with linear warmup and restarts.
    
    After the warmup phase, applies cosine annealing decay to min_scale. If the 
    decay cycle hasn't ended, the value stays at min_scale. When the cycle ends,
    the value resets to 1.0.
    """
    cycle_length = cycle_steps

    if step <= warmup_steps + cycle_steps:
        if step < warmup_steps:
            return linear_warmup(step, warmup_steps)
        else:
            step_in_cycle = step - warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_cycle / cycle_length))
            
            if min_scale is None:
                return 1.0 * cosine_decay
            else:
                return min_scale + (1.0 - min_scale) * cosine_decay
    else:
        first_cycle_length = cycle_steps + warmup_steps
        cycle_length = int(first_cycle_length * restarts_factor)
        step_in_cycle = step - first_cycle_length - ((step - first_cycle_length) // cycle_length) * cycle_length
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_in_cycle / (cycle_length - bench)))
        
        if min_scale is None:
            return decay_rate**((step - first_cycle_length) // cycle_length + 1) * 1.0 * cosine_decay
        else:
            if step_in_cycle < cycle_length - bench:
                return min_scale + ((1.0 - min_scale) * cosine_decay) * decay_rate**((step - first_cycle_length) // cycle_length + 1)
            else:
                return min_scale


def cosine_decay_with_exponential_tail_raw(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    min_scale: float = 0.2,
    exp_decay_rate: float = 0.65
) -> float:
    """
    Cosine decay to min_scale, then exponential decay from min_scale
    
    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        decay_steps: Number of cosine decay steps
        min_scale: Minimum scale at the end of cosine decay
        exp_decay_rate: Exponential decay rate
    """
    # Warmup phase
    if step < warmup_steps:
        return min(1.0, step / warmup_steps)
    
    # Adjust steps after warmup
    step = step - warmup_steps
    decay_steps = decay_steps - warmup_steps
    
    # Cosine decay phase
    if step < decay_steps:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (step / decay_steps)))
        return min_scale + (1.0 - min_scale) * cosine_decay
    
    # Exponential decay phase
    exp_decay = exp_decay_rate ** ((step - decay_steps) / decay_steps)
    return min_scale * exp_decay

def cosine_decay_with_exponential_tail(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    min_scale: float = 0.2,
    exp_decay_rate: float = 0.65,
    transition_start_ratio: float = 0.9,
    transition_length_ratio: float = 0.1
) -> float:
    """Computes the decay value using cosine decay and an exponential tail."""

    # Warmup phase
    if step < warmup_steps:
        return min(1.0, step / warmup_steps)
    
    # Adjust steps after warmup
    step = step - warmup_steps
    decay_steps = decay_steps - warmup_steps
    
    # Define the start and length of the transition
    transition_start = int(decay_steps * transition_start_ratio)
    # transition_length = int(decay_steps * transition_length_ratio)
    
    # Cosine decay phase
    if step < transition_start:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (step / decay_steps)))
        return min_scale + (1.0 - min_scale) * cosine_decay
    
    # Smooth transition phase
    # if step < (transition_start + transition_length):
    #     # Normalize position within the transition area
    #     transition_progress = (step - transition_start) / transition_length
        
    #     # Value at the start of the transition area (cosine)
    #     cosine_value = min_scale + (1.0 - min_scale) * (0.5 * (1 + math.cos(math.pi * (transition_start / decay_steps))))
        
    #     # Initial value of exponential decay
    #     start_exp_value = min_scale * (exp_decay_rate ** ((transition_start - decay_steps) / decay_steps))
        
    #     # Final value of exponential decay
    #     end_exp_value = min_scale * (exp_decay_rate ** (((step - decay_steps) / decay_steps)))

    #     # Use reverse quadratic interpolation for upward convexity
    #     transition_curve = 1 - (1 - transition_progress) ** 2
        
    #     # Interpolate between cosine and exponential values using start_exp_value
    #     interpolated_value = (1 - transition_curve) * cosine_value + transition_curve * (start_exp_value + (end_exp_value - start_exp_value) * transition_progress)
        
    #     return interpolated_value
    
    # Calculate the value at the start of the exponential decay phase
    cosine_value_at_transition_start = min_scale + (1.0 - min_scale) * (0.5 * (1 + math.cos(math.pi * (transition_start / decay_steps))))
    
    # Exponential decay phase
    exp_decay = exp_decay_rate ** ((step - transition_start) / decay_steps)
    return cosine_value_at_transition_start * exp_decay