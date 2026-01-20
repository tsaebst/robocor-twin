from __future__ import annotations
from typing import Dict, Any

def extract_world_state(env) -> Dict[str, Any]:
    """
    Extracts full 'world state' from RoboCourierEnv instance after reset.
    Requires direct attribute access (env.rx, env.px, ...).
    """
    return {
        "rx": int(env.rx), "ry": int(env.ry),
        "px": int(env.px), "py": int(env.py),
        "dx": int(env.dx), "dy": int(env.dy),
        "cx": int(env.cx), "cy": int(env.cy),
        "battery": int(env.battery),
        "has_package": bool(env.has_package),
    }
def force_world_state(env, state: Dict[str, Any]):
    """
    Forces RoboCourierEnv instance to a given state.
    """
    env.rx, env.ry = state["rx"], state["ry"]
    env.px, env.py = state["px"], state["py"]
    env.dx, env.dy = state["dx"], state["dy"]
    env.cx, env.cy = state["cx"], state["cy"]
    env.battery = state["battery"]
    env.has_package = state["has_package"]

