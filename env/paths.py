# env/paths.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import networkx as nx

from .network import NetworkState
from .nodes import UAV, GroundBS, Satellite, Node

K_PATHS = 4  # default number of candidate mesh paths per user

@dataclass
class Path:
    """候選路徑描述：GU -> access UAV -> (多跳 backhaul) -> GBS or Sat"""
    user_index: int             # GU index in state.gus
    access_uav_index: int       # UAV index in state.uavs
    backhaul_hops: List[Tuple[int, int]]  # (from_bs_idx, to_bs_idx) in state.bs_nodes
    uses_satellite: bool


def _bs_index_of_uav(state: NetworkState, uav_index: int) -> int:
    """UAV 在 bs_nodes 中的 index（UAVs 放在最前面）"""
    return uav_index  # 因為在 network.py 中: bs_nodes = uavs + gbss + sats


def _bs_index_of_gbs(state: NetworkState, gbs_index: int) -> int:
    """GBS 在 bs_nodes 中的 index"""
    return len(state.uavs) + gbs_index


def _bs_index_of_sat(state: NetworkState, sat_index: int) -> int:
    """Satellite 在 bs_nodes 中的 index"""
    return len(state.uavs) + len(state.gbss) + sat_index


def _build_backhaul_graph(state: NetworkState) -> nx.DiGraph:
    """
    用 backhaul capacity 建一個有向圖：
      node: bs_index
      edge: (i -> j)，權重為 1 / capacity，越大容量越便宜
    """
    num_bs = len(state.bs_nodes)
    G = nx.DiGraph()
    for i in range(num_bs):
        G.add_node(i)

    for i in range(num_bs):
        for j in range(num_bs):
            cap = state.bh_capacity[i, j]
            if cap <= 0:
                continue
            # cost 越小越好：容量越大 → cost 越小
            cost = 1.0 / cap
            G.add_edge(i, j, weight=cost, capacity=cap)

    return G


def _find_k_best_mesh_paths_for_uav(
    state: NetworkState,
    G: nx.DiGraph,
    bs_uav_idx: int,
    k_paths: int = 2,
) -> List[List[int]]:
    """
    從 access UAV 尋找至任意 GBS 的 K 條「好」路徑。
    使用 networkx.shortest_simple_paths，
    cost 為所有 edge weight (1/cap) 的總和（等價於偏好少 hop & 大容量）。
    """
    num_gbss = len(state.gbss)
    if num_gbss == 0:
        return []

    # 先收集所有 UAV->GBS 的最短路徑 candidate
    all_candidate_paths: List[Tuple[float, List[int]]] = []

    for gbs_idx in range(num_gbss):
        bs_gbs_idx = _bs_index_of_gbs(state, gbs_idx)
        try:
            # 取前幾條 simple paths
            gen = nx.shortest_simple_paths(G, bs_uav_idx, bs_gbs_idx, weight="weight")
            for path in gen:
                # path 是一串 bs_index，例如 [uav_bs_idx, ..., gbs_bs_idx]
                # 計算總 cost
                total_cost = 0.0
                for a, b in zip(path[:-1], path[1:]):
                    total_cost += G[a][b]["weight"]
                all_candidate_paths.append((total_cost, path))
                # 不要太多，先限制每個目的 GBS 收幾條 path
                if len(all_candidate_paths) >= 5 * k_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not all_candidate_paths:
        return []

    # 依 cost 排序，取前 k_paths 條不重複的路徑
    all_candidate_paths.sort(key=lambda x: x[0])
    selected_paths: List[List[int]] = []
    used_signatures = set()

    for _, path in all_candidate_paths:
        sig = tuple(path)
        if sig in used_signatures:
            continue
        used_signatures.add(sig)
        selected_paths.append(path)
        if len(selected_paths) >= k_paths:
            break

    return selected_paths


def build_candidate_paths_for_user(state: NetworkState, gu_idx: int) -> List[Path]:
    """
    為單一 user 建立候選路徑：
      - 多條 mesh path: GU -> access UAV -> (多跳 backhaul) -> 某個 GBS
      - Sat path (若存在): GU -> access UAV -> Sat（單跳）
    """
    paths: List[Path] = []

    num_uavs = len(state.uavs)
    num_gbss = len(state.gbss)
    num_sats = len(state.sats)

    if num_uavs == 0:
        return paths

    uav_idx = int(state.gu_to_uav[gu_idx])
    bs_uav_idx = _bs_index_of_uav(state, uav_idx)

    # ---------- 建立 backhaul graph ----------
    G = _build_backhaul_graph(state)

    # ---------- Mesh paths: 多跳 UAV/GBS ----------
    mesh_bs_paths = _find_k_best_mesh_paths_for_uav(state, G, bs_uav_idx, k_paths=K_PATHS)

    for bs_path in mesh_bs_paths:
        # bs_path 例如 [uav_bs_idx, ..., gbs_bs_idx]
        hops: List[Tuple[int, int]] = []
        for a, b in zip(bs_path[:-1], bs_path[1:]):
            hops.append((a, b))
        if not hops:
            continue

        mesh_path = Path(
            user_index=gu_idx,
            access_uav_index=uav_idx,
            backhaul_hops=hops,
            uses_satellite=False,
        )
        paths.append(mesh_path)

    # ---------- Satellite path: UAV -> Sat (單跳，若存在 & 有 capacity) ----------
    if num_sats > 0:
        sat_bs_idx = _bs_index_of_sat(state, 0)  # 目前假設只有一顆
        cap = state.bh_capacity[bs_uav_idx, sat_bs_idx]
        if cap > 0:
            sat_path = Path(
                user_index=gu_idx,
                access_uav_index=uav_idx,
                backhaul_hops=[(bs_uav_idx, sat_bs_idx)],
                uses_satellite=True,
            )
            paths.append(sat_path)

    return paths


def build_all_candidate_paths(state: NetworkState) -> Dict[int, List[Path]]:
    """
    為所有 users 建立候選路徑：
    回傳 dict: user_index -> List[Path]
    """
    num_gus = len(state.gus)
    user_paths: Dict[int, List[Path]] = {}
    for gu_idx in range(num_gus):
        user_paths[gu_idx] = build_candidate_paths_for_user(state, gu_idx)
    return user_paths
