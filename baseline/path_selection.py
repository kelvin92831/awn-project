# baseline/path_selection.py
from __future__ import annotations

from typing import Dict, List, Optional

from env.network import NetworkState
from env.paths import Path


def select_paths_baseline(
    state: NetworkState,
    candidate_paths: Dict[int, List[Path]],
) -> Dict[int, Optional[Path]]:
    """
    Baseline 路徑選擇策略：
      - 若有 mesh path（uses_satellite=False），優先選 mesh
      - 否則若只有 sat path，選 sat
      - 若沒有候選路徑，記為 None（之後 rate=0）
    回傳 dict: user_index -> Path or None
    """
    chosen: Dict[int, Optional[Path]] = {}
    num_gus = len(state.gus)

    for gu_idx in range(num_gus):
        paths = candidate_paths.get(gu_idx, [])
        if not paths:
            chosen[gu_idx] = None
            continue

        # 優先 mesh
        mesh_paths = [p for p in paths if not p.uses_satellite]
        if mesh_paths:
            chosen[gu_idx] = mesh_paths[0]
        else:
            # 沒有 mesh，就選第一個（通常是 sat）
            chosen[gu_idx] = paths[0]

    return chosen
