"""Environment adapters.

Each adapter implements the `Env` interface: produce tasks, run an agent against a
task, and return a Trajectory + verifier verdict.

  - ToyEnv          deterministic mini-website used by tests and offline examples
  - WebArenaEnv     wraps the WebArena docker stack (Playwright under the hood)
  - VisualWebArenaEnv similar, with extra screenshot-based verifiers
  - OnlineMind2WebEnv  live-web adapter with the 9/3 domain split
"""

from scaffold.envs.base import Env, Task, RolloutConfig
from scaffold.envs.toy_env import ToyEnv, ToyTask, ToyBrowser
from scaffold.envs.webarena_env import WebArenaEnv
from scaffold.envs.vwa_env import VisualWebArenaEnv
from scaffold.envs.om2w_env import OnlineMind2WebEnv

__all__ = [
    "Env",
    "Task",
    "RolloutConfig",
    "ToyEnv",
    "ToyTask",
    "ToyBrowser",
    "WebArenaEnv",
    "VisualWebArenaEnv",
    "OnlineMind2WebEnv",
]
