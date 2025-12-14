# simulations/debug_stage01.py
from env.network import NetworkEnv
from env.paths import build_all_candidate_paths


def main():
    env = NetworkEnv(seed=42)
    state = env.init_random_state()

    print(f"#GUs = {len(state.gus)}, #UAVs = {len(state.uavs)}, "
          f"#GBSs = {len(state.gbss)}, #Sats = {len(state.sats)}")
    print("Backhaul capacity matrix shape:", state.bh_capacity.shape)

    all_paths = build_all_candidate_paths(state)
    # 隨便印前 5 個 user 的候選路徑
    for gu_idx in range(min(80, len(state.gus))):
        paths = all_paths[gu_idx]
        gu = state.gus[gu_idx]
        print(f"\nUser {gu_idx} (urgent={gu.is_urgent}):")
        if not paths:
            print("  No candidate paths.")
            continue
        for p in paths:
            hop_str = " -> ".join([f"{frm}->{to}" for frm, to in p.backhaul_hops])
            print(f"  uses_sat={p.uses_satellite}, access_uav={p.access_uav_index}, "
                  f"backhaul_hops: {hop_str}")


if __name__ == "__main__":
    main()
