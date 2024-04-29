"""
Configure and Model Mappings

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""

from .adamole import AdaMoleConfig, AdaMoleModel
from .lora import LoraConfig, LoraModel
from .mole import MoleConfig, MoleModel
from .utils.peft_types import PeftType

PEFT_TYPE_TO_CONFIG_MAPPING = {
    PeftType.LORA: LoraConfig,
    PeftType.MOLE: MoleConfig,
    PeftType.ADAMOLE: AdaMoleConfig,
}
PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.MOLE: MoleModel,
    PeftType.ADAMOLE: AdaMoleModel,
}
