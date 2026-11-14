"""與 Isaac Sim 解耦的河內塔規劃與數值輔助函式。

本模組只產生符號移動序列及純數值結果，不建立 USD 物件、不讀取剛體狀態，
也不直接驅動機械手臂。`tower.py` 會消費 `plan_hanoi()` 的移動序列；
高度計算和插值函式則提供可獨立測試的規劃工具。

移動中的盤子編號由 0 起算，0 是最小盤；柱位編號 0、1、2 分別對應
起始柱、輔助柱、目標柱。N 個盤子的遞迴解需要 `2**N - 1` 個合法移動。
"""

from __future__ import annotations

# 移動步驟元組定義：(盤子編號, 來源柱編號, 目標柱編號)
# 盤子編號：0 為頂層最小盤，數字越大盤子越大
# 柱子編號：0 (起始柱), 1 (輔助柱), 2 (目標柱)
Move = tuple[int, int, int]


def smoothstep_fraction(step: int, total_steps: int) -> float:
    """回傳三次 Hermite smoothstep 的正規化插值比例。

    `t` 會依 `int(total_steps * 0.75)` 正規化並限制在 [0, 1]；因此最後約
    四分之一的呼叫步數會維持在終點比例 1。多項式 `3t² - 2t³` 在兩端
    斜率為 0，可用於平滑開始與停止。這是獨立輔助函式；目前 FSM 使用
    `FrankaPickPlace._smoothstep()` 產生各階段軌跡。

    Args:
        step: 目前的零起算步數。
        total_steps: 計畫步數，必須大於 0。

    Returns:
        介於 0 與 1 的插值比例。

    Raises:
        ValueError: `total_steps` 小於或等於 0。
    """
    if total_steps <= 0:
        raise ValueError("total_steps 必須為正整數")
    motion_steps = max(1, int(total_steps * 0.75))
    progress = min(max(step / motion_steps, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def smooth_velocity_scale(step: int, total_steps: int) -> float:
    """計算由 0.2 平滑升至 1.0 的比例係數。

    前 20% 的階段進度以 smoothstep 漸增，之後維持 1.0。此函式目前供
    獨立數值測試與後續控制器調參使用；主 FSM 不會呼叫它。

    Args:
        step: 目前的零起算步數。
        total_steps: 計畫步數，必須大於 0。

    Returns:
        介於 0.2 與 1.0 的縮放比例。

    Raises:
        ValueError: `total_steps` 小於或等於 0。
    """
    if total_steps <= 0:
        raise ValueError("total_steps 必須為正整數")
    progress = min(max((step + 1) / total_steps, 0.0), 1.0)
    ramp_progress = min(progress / 0.2, 1.0)
    return 0.2 + 0.8 * ramp_progress * ramp_progress * (3.0 - 2.0 * ramp_progress)


def required_lift_height(
    current_disk_center_z: float,
    stack_layers: int,
    block_height: float = 0.03,
    reserve: float = 0.08,
    minimum: float = 0.12,
) -> float:
    """估算水平搬運前所需的 Z 軸抬升量（回傳位移量，不是絕對高度）。

    目標盤中心設在最高堆疊頂面之上，額外預留半個盤厚及安全間距：
    `target_z = stack_layers * block_height + block_height / 2 + reserve`。
    回傳值是 `target_z - current_disk_center_z` 與 `minimum` 的較大者。
    呼叫端仍須將此位移量套用到手臂目標，並以實際碰撞幾何驗證安全性。

    參數：
        current_disk_center_z: 目前盤子中心的世界 Z 座標（公尺）。
        stack_layers: 路徑上最高堆疊的層數。
        block_height: 單一盤子的厚度（公尺）。
        reserve: 超過堆疊頂面的預留距離（公尺）。
        minimum: 最小抬升位移（公尺）。
    返回：
        手臂末端需要向上移動的距離（公尺）。
    """
    target_disk_center_z = stack_layers * block_height + reserve + block_height / 2
    return max(minimum, target_disk_center_z - current_disk_center_z)


def plan_hanoi(disk_count: int) -> list[Move]:
    """產生從柱 0 搬到柱 2 的最少合法移動序列。

    對 `count` 個盤子，遞迴分解為：先將上方 `count - 1` 個盤子移到輔助柱，
    再移動目前最大的盤，最後把輔助柱上的小盤移到目標柱。每個輸出 tuple
    為 `(disk_index, source_peg, destination_peg)`；本函式只保證演算法序列，
    真正的剛體位置與移動合法性由 `tower.py` 的物理控制流程負責。

    Args:
        disk_count: 盤子數量，必須至少為 1。

    Returns:
        依執行順序排列的移動 tuple 清單，長度為 `2**disk_count - 1`。

    Raises:
        ValueError: `disk_count` 小於 1。
    """
    if disk_count < 1:
        raise ValueError("disk_count 至少為 1")

    moves: list[Move] = []

    def move_stack(count: int, source: int, destination: int, spare: int) -> None:
        # 遞迴不變條件：把 source 柱最上方 count 個盤子，依合法大小順序搬到 destination；
        # spare 是唯一可暫放的柱子。呼叫返回時，這 count 個盤子已全部在 destination。
        if count == 0:
            # 沒有盤子需要搬，是遞迴終止條件；不產生任何移動。
            return

        # 步驟一：先搬走壓在最大盤上方的 count-1 個小盤。
        # 此子問題的目的地改成 spare，而原 destination 暫時充當新的 spare。
        move_stack(count - 1, source, spare, destination)

        # 步驟二：此時 source 頂端已是這組盤子中最大的盤（索引 count-1），
        # destination 為空或頂盤比它大，所以可以合法地直接移動。
        moves.append((count - 1, source, destination))

        # 步驟三：把剛才暫放在 spare 的 count-1 個小盤搬到最大盤上。
        # 這次 destination 是原目標柱，source 則成為可暫放小盤的 spare。
        move_stack(count - 1, spare, destination, source)

    # 由小子問題組合大問題：每層需完成兩個 count-1 子問題，再移動一次最大盤，
    # 因此 T(n)=2T(n-1)+1、T(0)=0，總步數為 2**n-1，亦為三柱 Hanoi 的最少步數。
    # 初始任務是由 Peg 0 搬到 Peg 2，Peg 1 作為暫存柱。
    move_stack(disk_count, 0, 2, 1)
    return moves
