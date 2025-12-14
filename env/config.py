# env/config.py
import math

# -------------------------
# 區域設定（單位：公尺）
# -------------------------
AREA_X_MIN = 0.0
AREA_X_MAX = 2000.0  # 2 km
AREA_Y_MIN = 0.0
AREA_Y_MAX = 2000.0  # 2 km

# -------------------------
# 節點數量
# -------------------------
NUM_GUS = 80     # Ground Users
NUM_UAVS = 4     # UAV relays
NUM_GBSS = 1     # Ground BS
NUM_SATS = 1     # Satellites (先用 1 顆)

# 高度（m）
UAV_HEIGHT = 150.0
GBS_HEIGHT = 30.0
SAT_HEIGHT = 160_000.0  # 簡化版：低軌衛星高度（可再調整） #160 km

# -------------------------
# 通訊設定
# -------------------------
C = 3e8  # 光速
CARRIER_FREQ = 2.4e9     # 2.4 GHz
BANDWIDTH_HZ = 10e6      # 10 MHz 系統頻寬（只用來算容量）

TX_POWER_DBM = 30.0      # 1 W
TX_POWER_WATT = 10 ** (TX_POWER_DBM / 10.0) / 1000.0

THERMAL_NOISE_DBM_PER_HZ = -174  # dBm/Hz
NOISE_FIGURE_DB = 9.0

# 只保留 SNR 高於此門檻的 backhaul link
MIN_SNR_DB_BACKHAUL = -20.0 # -5 dB

# -------------------------
# 路徑損耗參數（簡化）
# -------------------------
PATH_LOSS_EXP_A2A = 2.2  # UAV-UAV / UAV-GBS
PATH_LOSS_EXP_G2G = 3.0  # GBS-GBS 或地面對地面
PATH_LOSS_EXP_SAT = 2.0  # 衛星相關（簡化成接近自由空間）

# A2G LoS / NLoS model
ETA_LOS_DB = 1.0
ETA_NLOS_DB = 20.0
B1 = 9.61
B2 = 0.16


def calc_noise_power_watt() -> float:
    """計算在 BANDWIDTH_HZ 下的雜訊功率（瓦特）"""
    noise_dbm = (
        THERMAL_NOISE_DBM_PER_HZ
        + 10 * math.log10(BANDWIDTH_HZ)
        + NOISE_FIGURE_DB
    )
    noise_mw = 10 ** (noise_dbm / 10.0)
    return noise_mw / 1000.0


NOISE_POWER_WATT = calc_noise_power_watt()


# -------------------------
# Traffic / QoS 相關設定
# -------------------------

# Urgent 使用者比例（例如 0.2 表示 20% 使用者是 urgent）
URGENT_USER_RATIO = 0.2

# 要求速率（bps）
REQ_RATE_NORMAL_BPS = 2e6   # 一般使用者要求 2 Mbps
REQ_RATE_URGENT_BPS = 3e6   # Urgent 使用者要求 3 Mbps

# Sigmoid steepness
V_NORMAL = 1.0
V_URGENT = 2.0
