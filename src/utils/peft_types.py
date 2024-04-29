"""
PEFT and Task Types

Portions of this file are modifications based on work created and
shared by the HuggingFace Inc. team and used according to terms
described in the Apache License 2.0.
"""
import enum


class PeftType(str, enum.Enum):
    """
    PEFT Adapter Types
    """
    LORA = "LORA"
    MOLE = "MOLE"
    ADAMOLE = "ADAMOLE"


class TaskType(str, enum.Enum):
    """
    PEFT Task Type
    """
    CAUSAL_LM = "CAUSAL_LM"
