"""VPP 구성: 지리 층 반클러스터링."""

from src.oibc.compose.anticluster import balanced_split, build_composition, spatial_strata
from src.oibc.compose.baselines import capacity_balanced_assignment, random_assignment
from src.oibc.compose.constraints import CompositionError, validate_assignment
from src.oibc.compose.store import CompositionRecord, assignment_sha256, load_composition, write_composition

__all__ = [
    "CompositionError",
    "CompositionRecord",
    "assignment_sha256",
    "balanced_split",
    "build_composition",
    "capacity_balanced_assignment",
    "load_composition",
    "random_assignment",
    "spatial_strata",
    "validate_assignment",
    "write_composition",
]
