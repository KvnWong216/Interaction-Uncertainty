"""Prompt-conditioned uncertainty-to-interaction action bridge.

The package intentionally keeps perception models, VLA backbones, simulators,
and robot SDKs behind small protocols.  The mathematical core is CPU-only.
"""

from .beliefs import BetaBelief, DirichletBelief, EvidenceDeficit, TaskBelief
from .updates import project_prompt_target_beta, update_beta, update_dirichlet
from .v2 import (
    CandidateEffectForecast,
    CandidateSet,
    EpisodeController,
    EvidencePacket,
    InformationNeed,
    PlanningWeights,
    PrimitiveCall,
    PrimitiveDecision,
    PrimitiveKind,
    TaskSpec,
)

__all__ = [
    "BetaBelief",
    "CandidateEffectForecast",
    "CandidateSet",
    "DirichletBelief",
    "EvidenceDeficit",
    "EpisodeController",
    "EvidencePacket",
    "InformationNeed",
    "PlanningWeights",
    "PrimitiveCall",
    "PrimitiveDecision",
    "PrimitiveKind",
    "TaskBelief",
    "TaskSpec",
    "project_prompt_target_beta",
    "update_beta",
    "update_dirichlet",
]

__version__ = "0.2.0"
