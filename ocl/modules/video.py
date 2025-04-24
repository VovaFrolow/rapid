from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import nn

from ocl.utils import make_build_fn
from ocl.modules.utils import WNConv3d


@make_build_fn(__name__, "video module")
def build(config, name: str, **kwargs):
    pass  # No special module building needed


class LatentProcessor(nn.Module):
    """Updates latent state based on inputs and state and predicts next state."""

    def __init__(
        self,
        corrector: nn.Module,
        predictor: Optional[nn.Module] = None,
        state_key: str = "slots",
        first_step_corrector_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.corrector = corrector
        self.predictor = predictor
        self.state_key = state_key
        if first_step_corrector_args is not None:
            self.first_step_corrector_args = first_step_corrector_args
        else:
            self.first_step_corrector_args = None

    def forward(
        self, state: torch.Tensor, inputs: Optional[torch.Tensor], time_step: Optional[int] = None, 
        prev_state: Optional[torch.Tensor] = None, mode: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        # state: batch x n_slots x slot_dim
        assert state.ndim == 3
        # inputs: batch x n_inputs x input_dim
        assert inputs.ndim == 3
        # memory = prev_state
        if inputs is not None:
            if mode != "tsmm":
                if time_step == 0 and self.first_step_corrector_args:
                    corrector_output = self.corrector(state, inputs, **self.first_step_corrector_args)
                else:
                    corrector_output = self.corrector(state, inputs)
                    # corrector_output = self.corrector(state, inputs)
            else:
                if time_step == 0 and self.first_step_corrector_args:
                    corrector_output = self.corrector(state, inputs, **self.first_step_corrector_args, prev_state=prev_state)
                else:
                    corrector_output = self.corrector(state, inputs, prev_state=prev_state)
                # memory = corrector_output[self.state_key]
            updated_state = corrector_output[self.state_key]
        else:
            # Run predictor without updating on current inputs
            corrector_output = None
            updated_state = state

        if self.predictor:
            predicted_state = self.predictor(updated_state)
        else:
            # Just pass updated_state along as prediction
            predicted_state = updated_state

        return {
            "state": updated_state,
            "state_predicted": predicted_state,
            "corrector": corrector_output,
            # "memory": memory if mode == "tsmm" else None,
        }


class MapOverTime(nn.Module):
    """Wrapper applying wrapped module independently to each time step.

    Assumes batch is first dimension, time is second dimension.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args):
        batch_size = None
        seq_len = None
        flattened_args = []
        for idx, arg in enumerate(args):
            B, T = arg.shape[:2]
            if not batch_size:
                batch_size = B
            elif batch_size != B:
                raise ValueError(
                    f"Inconsistent batch size of {B} of argument {idx}, was {batch_size} before."
                )

            if not seq_len:
                seq_len = T
            elif seq_len != T:
                raise ValueError(
                    f"Inconsistent sequence length of {T} of argument {idx}, was {seq_len} before."
                )

            flattened_args.append(arg.flatten(0, 1))

        outputs = self.module(*flattened_args)

        if isinstance(outputs, Mapping):
            unflattened_outputs = {
                k: v.unflatten(0, (batch_size, seq_len)) for k, v in outputs.items()
            }
        else:
            unflattened_outputs = outputs.unflatten(0, (batch_size, seq_len))

        return unflattened_outputs


class ScanOverTime(nn.Module):
    """Wrapper applying wrapped module recurrently over time steps"""

    def __init__(
        self, module: nn.Module, next_state_key: str = "state_predicted", 
        state_key: str = "state", pass_step: bool = True, mode: Optional[str] = None
    ) -> None:
        super().__init__()
        self.module = module
        self.state_key = state_key
        self.next_state_key = next_state_key
        self.pass_step = pass_step
        if mode == "tsmm":
            self.state_size = 2 * module.corrector.dim
        self.mode = mode

    def forward(self, initial_state: torch.Tensor, inputs: torch.Tensor):
        # initial_state: batch x ...
        # inputs: batch x n_frames x ...
        b, seq_len, *_ = inputs.shape
        # seq_len = inputs.shape[1]
        state = initial_state
        if self.mode == "tsmm":
            prev_state = None
        outputs = []
        for t in range(seq_len):
            if self.mode != "tsmm":
                if self.pass_step:
                    output = self.module(state, inputs[:, t], t)
                else:
                    output = self.module(state, inputs[:, t])
                state = output[self.next_state_key]
            else:
                if self.pass_step:
                    output = self.module(state, inputs[:, t], t, prev_state=prev_state, mode=self.mode)
                else:
                    output = self.module(state, inputs[:, t], prev_state=prev_state, mode=self.mode)
                
                # if (t + 1) % 6 == 0:  # Truncated BPTT
                #     state = output[self.next_state_key].detach()
                # else:
                state = output[self.next_state_key]
                prev_state = output[self.state_key]
            outputs.append(output)
        
        return merge_dict_trees(
            outputs, 
            axis=1, 
            tconv=self.module.corrector.knconv if isinstance(self.module.corrector.knconv, WNConv3d) else None
        )


def merge_dict_trees(trees: List[Mapping], axis: int = 0, tconv: Optional[nn.Module] = None):
    """Stack all leafs given a list of dictionaries trees.

    Example:
    x = merge_dict_trees([
        {
            "a": torch.ones(2, 1),
            "b": {"x": torch.ones(2, 2)}
        },
        {
            "a": torch.ones(3, 1),
            "b": {"x": torch.ones(1, 2)}
        }
    ])

    x == {
        "a": torch.ones(5, 1),
        "b": {"x": torch.ones(3, 2)}
    }
    """
    out = {}
    if len(trees) > 0:
        ref_tree = trees[0]
        for key, value in ref_tree.items():
            values = [tree[key] for tree in trees]
            if isinstance(value, torch.Tensor):
                out[key] = torch.stack(values, axis)
            elif isinstance(value, Mapping):
                out[key] = merge_dict_trees(values, axis)
            else:
                out[key] = values
    
    if tconv is not None:
        att_maps = out["corrector"]["masks"]
        b, t, k, d = att_maps.shape
        att_maps = att_maps.permute(0, 2, 1, 3).unsqueeze(2).reshape(b*k, 1, t, int(d**0.5), int(d**0.5))
        att_maps = tconv(att_maps)
        out["corrector"]["masks"] = att_maps.squeeze().reshape(b, k, t, d).permute(0, 2, 1, 3)

    return out
