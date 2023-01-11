from functools import partial
# from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from oilcat.oil_env import OilCat

# def env_fn(env, **kwargs) -> MultiAgentEnv:
#     return env(**kwargs)


# REGISTRY = {
#     "sc2": partial(env_fn, env=OilCatEnv),
# }

REGISTRY = {
    "sc2": OilCat,
}

# if sys.platform == "linux":
#     os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
