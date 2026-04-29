from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    repo_sc2_path = os.path.join(repo_root, "3rdparty", "StarCraftII")
    os.environ.setdefault("SC2PATH", repo_sc2_path)
