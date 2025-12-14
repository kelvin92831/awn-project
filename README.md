# AI-Assisted Cross-Layer Routing and Resource Allocation for Disaster-Resilient Terrestrial–UAV–Satellite Networks

This project simulates a heterogeneous wireless backhaul network (involving UAVs, Satellites, and Ground Stations) to evaluate and compare different path selection and bandwidth allocation schemes.

## Project Structure

- **`ai/`**: Contains the AI-assisted path selection logic using a feature-based linear model (`theta · phi`) with congestion awareness.
- **`baseline/`**: Implements baseline heuristic algorithms for path selection and bandwidth allocation (e.g., Equal Share, Priority-Weighted).
- **`env/`**: Defines the simulation environment, including:
    - `network.py`: Network state and topology management.
    - `channel.py`: Channel models (SNR, capacity).
    - `nodes.py`: User and Base Station definitions.
    - `paths.py`: Graph construction and path finding.
- **`simulations/`**: Scripts for running experiments and generating results.
- **`figures/`**: Directory where generated plots are saved.

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Simulations

To run the full suite of scenarios and collect metrics for all schemes (Equal Share, Priority, AI):

```bash
python -m simulations.run_scenarios_all_schemes
```

### Generating Figures

To generate the performance comparison plots (saved to `figures/`):

```bash
python -m simulations.plot_figures
```

This will produce:
- `fig1_weighted_sum_utility.png`: Comparison of weighted sum utility.
- `fig2a_coverage_tradeoff.png`: Coverage ratio for urgent vs. non-urgent users.
- `fig2b_utility_tradeoff.png`: Average utility for urgent vs. non-urgent users.
- `fig4_kpaths_effect.png`: Impact of the number of candidate paths (`k_paths`) on performance.

## Methods Compared

1.  **Equal-Share Baseline**:
    -   Standard routing.
    -   Backhaul bandwidth is shared equally among users.

2.  **Priority-Weighted Baseline**:
    -   Standard routing.
    -   Backhaul bandwidth is allocated based on user weights (prioritizing urgent traffic).

3.  **AI-Assisted**:
    -   Feature-based path selection considering congestion and path characteristics.
    -   Priority-weighted bandwidth allocation.

