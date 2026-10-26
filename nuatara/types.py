from dataclasses import dataclass
from typing import Optional

@dataclass
class BackendInfo:
    backend_name: str
    device: str
    precision: str
    sdk_version: Optional[str] = None
    model_path: Optional[str] = None
    provider: Optional[str] = None
