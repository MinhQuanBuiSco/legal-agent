from dataclasses import dataclass, field
from typing import Optional


class DataConfig:
    """Data configuration."""

    dataset_name: str = "HFforLegal/case-law"