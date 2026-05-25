"""SCAFFOLD: Self-Improving Web Agents via Recursive Parametric Skill Abstraction.

Top-level imports expose the most common entry points; for deep customization the
sub-packages should be imported directly.
"""

from scaffold.core.skill import Skill, SkillCall, Parameter
from scaffold.core.library import SkillLibrary
from scaffold.core.trajectory import Trajectory, Observation, Action
from scaffold.pipeline import ScaffoldPipeline, ScaffoldConfig

__version__ = "0.1.0"

__all__ = [
    "Skill",
    "SkillCall",
    "Parameter",
    "SkillLibrary",
    "Trajectory",
    "Observation",
    "Action",
    "ScaffoldPipeline",
    "ScaffoldConfig",
]
