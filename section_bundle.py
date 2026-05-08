import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class SectionBundle:
    bundle_path: Path
    section_indices_path: Optional[Path] = None
    friction_mus: Dict[int, float] = field(default_factory=dict)
    speed_scales: Dict[int, float] = field(default_factory=dict)
    target_speeds: Dict[int, float] = field(default_factory=dict)
    approach_speeds: Dict[int, float] = field(default_factory=dict)
    approach_leads: Dict[int, float] = field(default_factory=dict)
    lookahead_scales: Dict[int, float] = field(default_factory=dict)
    steer_gain_scales: Dict[int, float] = field(default_factory=dict)
    brake_distance_scales: Dict[int, float] = field(default_factory=dict)
    direct_target_sections: Set[int] = field(default_factory=set)


def _normalize_float_map(values) -> Dict[int, float]:
    if not isinstance(values, dict):
        return {}
    return {int(k): float(v) for k, v in values.items()}


def _resolve_optional_path(bundle_path: Path, raw_value: Optional[str]) -> Optional[Path]:
    if not raw_value:
        return None
    raw_path = Path(str(raw_value))
    if raw_path.is_absolute():
        return raw_path
    if raw_path.exists():
        return raw_path.resolve()
    rel = bundle_path.parent / raw_path
    if rel.exists():
        return rel.resolve()
    return rel


def load_section_bundle(path: str) -> SectionBundle:
    bundle_path = Path(path).resolve()
    with bundle_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raw = {}
    globals_map = raw.get("globals", {})
    if not isinstance(globals_map, dict):
        globals_map = {}

    direct_raw = globals_map.get("section_direct_target_sections", [])
    if isinstance(direct_raw, str):
        direct_sections = {int(v.strip()) for v in direct_raw.split(",") if v.strip()}
    elif isinstance(direct_raw, list):
        direct_sections = {int(v) for v in direct_raw}
    else:
        direct_sections = set()

    return SectionBundle(
        bundle_path=bundle_path,
        section_indices_path=_resolve_optional_path(
            bundle_path,
            globals_map.get("competition_section_indices_json"),
        ),
        friction_mus=_normalize_float_map(raw.get("friction_mus", raw.get("friction_mu", {}))),
        speed_scales=_normalize_float_map(raw.get("speed_scales", {})),
        target_speeds=_normalize_float_map(raw.get("target_speeds", {})),
        approach_speeds=_normalize_float_map(raw.get("approach_speeds", {})),
        approach_leads=_normalize_float_map(raw.get("approach_leads", {})),
        lookahead_scales=_normalize_float_map(raw.get("lookahead_scales", {})),
        steer_gain_scales=_normalize_float_map(raw.get("steer_gain_scales", {})),
        brake_distance_scales=_normalize_float_map(raw.get("brake_distance_scales", {})),
        direct_target_sections=direct_sections,
    )


def load_section_indices(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return []
    return [int(v) for _k, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
