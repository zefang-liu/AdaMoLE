"""
MoLE Configuration
"""
from dataclasses import dataclass, field

from ..lora import LoraConfig
from ..utils.peft_types import PeftType


@dataclass
class MoleConfig(LoraConfig):
    """
    MoLE Configuration
    """
    num_experts: int = field(default=4, metadata={"help": "The number of experts in MoE."})
    top_k: int = field(default=None, metadata={
        "help": "The k in top-k gating if the expert threshold is None."})
    threshold: float = field(default=None, metadata={
        "help": "The threshold for selecting experts if the top-k is None. "
                "The maximum threshold should be 1 / number of experts"})

    def __post_init__(self):
        self.peft_type = PeftType.MOLE
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
