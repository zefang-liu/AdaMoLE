"""
AdaMoLE Model
"""
from typing import Any

import torch
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from .config import AdaMoleConfig
from .layer import AdaMoleLayer, LinearAdaMoleLayer
from ..lora import LoraModel


class AdaMoleModel(LoraModel):
    """
    AdaMoLE (Adaptive Mixture of LoRA Experts) Model
    """
    prefix: str = "lora_"

    def __init__(self, model: nn.Module, config: AdaMoleConfig, adapter_name: str = "default") -> None:
        super().__init__(model, config, adapter_name)

    def _create_and_replace(
        self, adamole_config: AdaMoleConfig, adapter_name: str,
        target: nn.Module, target_name: str, parent: nn.Module, **kwargs: Any,
    ) -> None:
        """
        Inplace replacement of the target module with the adapter layer
        """
        kwargs = {
            "lora_rank": adamole_config.lora_rank,
            "lora_alpha": adamole_config.lora_alpha,
            "lora_dropout": adamole_config.lora_dropout,
            "init_lora_weights": adamole_config.init_lora_weights,
            "num_experts": adamole_config.num_experts,
            "max_threshold": adamole_config.max_threshold,
        }

        if isinstance(target, AdaMoleLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(adapter_name: str, target: nn.Module, **kwargs: Any) -> nn.Module:
        """
        Create the new LoRA module for the target module
        """
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = LinearAdaMoleLayer(base_layer=target, adapter_name=adapter_name, **kwargs)
        else:
            raise ValueError(
                f"The target module `{target}` is not supported. "
                f"Currently, only the following modules are supported: `torch.nn.Linear`.")

        return new_module

    def get_aux_loss(self, adapter_name="default") -> torch.Tensor:
        """
        Get the load balancing loss for the whole model
        """
        model_loss = torch.tensor(0, dtype=torch.float).to(self.model.device)

        for name, module in self.model.named_modules():
            if name.endswith('moe_layer'):
                layer_loss = module[adapter_name].layer_loss
                model_loss += layer_loss

        return model_loss
