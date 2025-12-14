# env/nodes.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Node:
    """通用節點類別（GU / UAV / GBS / Satellite）"""
    id: int
    x: float
    y: float
    z: float

    @property
    def pos(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass
class GroundUser(Node):
    """地面使用者"""
    is_urgent: bool = False
    r_req: float = 1e6   # 要求速率（預設 1 Mbps）
    v_s: float = 1.0     # QoS sigmoidal steepness


@dataclass
class UAV(Node):
    """UAV 中繼節點"""
    pass


@dataclass
class GroundBS(Node):
    """地面基地台"""
    pass


@dataclass
class Satellite(Node):
    """衛星閘道（或其在投影平面上的等效位置）"""
    pass
