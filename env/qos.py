# env/qos.py
import math
from .nodes import GroundUser


def user_utility(user: GroundUser, rate_bps: float) -> float:
    """
    Sigmoid QoS utility:
        U_s = 1 / (1 + exp(-v_s * (rate / r_req - 1)))
    rate_bps, r_req 均為 bps。
    """
    r_req = max(user.r_req, 1e-6)
    v_s = user.v_s
    x = rate_bps / r_req - 1.0
    return 1.0 / (1.0 + math.exp(-v_s * x))


def user_weight(user: GroundUser) -> float:
    """
    Priority weight w_s：urgent flow 給較高權重。
    之後若你想更細緻，可以改這裡。
    """
    return 2.0 if user.is_urgent else 1.0
