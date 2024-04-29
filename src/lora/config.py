"""
LoRA Configuration

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

from ..config import PeftConfig
from ..utils.peft_types import PeftType


@dataclass
class LoraConfig(PeftConfig):
    """
    LoRA Configuration
    """
    lora_rank: int = field(default=8, metadata={"help": "The Lora rank for the attention dimension."})
    lora_alpha: int = field(default=8, metadata={"help": "The alpha parameter for Lora scaling."})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for Lora layers."})
    bias: Literal["none", "all", "lora_only"] = field(
        default="none", metadata={"help": "The bias type for Lora layers and can be 'none', 'all' or 'lora_only'."})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None, metadata={"help": "The names of the modules to apply the adapter to."})
    init_lora_weights: bool = field(
        default=True, metadata={"help": "Whether to initialize the weights of the adapter layers."})

    def __post_init__(self):
        self.peft_type = PeftType.LORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
