"""與 Isaac Sim 解耦的河內塔規劃與數值輔助函式。

本模組只產生符號移動序列及純數值結果，不建立 USD 物件、不讀取剛體狀態，
也不直接驅動機械手臂。tower.py 會消費 plan_hanoi() 的移動序列；
高度計算和插值函式則提供可獨立測試的規劃工具。

移動中的盤子索引由 0 起算，索引 0 對應最小盤 D1，索引 i 對應 Di+1，
所以索引 N-1 對應最大盤 DN。柱位編號 0、1、2 分別對應起始柱、輔助柱、
目標柱。N 個盤子的遞迴解需要 2**N - 1 個合法移動。
"""

from __future__ import annotations

from collections import deque

# 移動步驟元組定義：(盤子索引, 來源柱編號, 目標柱編號)
# 盤子索引 0 對應 D1（最小盤），索引 i 對應 D(i+1)，數字越大盤子越大
# 柱子編號：0 (起始柱), 1 (輔助柱), 2 (目標柱)
Move = tuple[int, int, int]

# 狀態型別：三根柱子各自的盤子堆疊由底至頂（不可變 tuple）
_State = tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


def smoothstep_fraction(step: int, total_steps: int) -> float:
    """回傳三次 Hermite smoothstep 的正規化插值比例。

    t 會依 int(total_steps * 0.75) 正規化並限制在 [0, 1]；因此最後約
    四分之一的呼叫步數會維持在終點比例 1。多項式 3t² - 2t³ 在兩端
    斜率為 0，可用於平滑開始與停止。這是獨立輔助函式；目前 FSM 使用
    FrankaPickPlace._smoothstep() 產生各階段軌跡。

    Args:
        step: 目前的零起算步數。
        total_steps: 計畫步數，必須大於 0。

    Returns:
        介於 0 與 1 的插值比例。

    Raises:
        ValueError: total_steps 小於或等於 0。
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
        ValueError: total_steps 小於或等於 0。
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
    target_z = stack_layers * block_height + block_height / 2 + reserve。
    回傳值是 target_z - current_disk_center_z 與 minimum 的較大者。
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

    對 count 個盤子，遞迴分解為：先將上方 count - 1 個盤子移到輔助柱，
    再移動目前最大的盤，最後把輔助柱上的小盤移到目標柱。每個輸出 tuple
    為 (disk_index, source_peg, destination_peg)，其中 disk_index=0 對應 D1，
    disk_index=i 對應 D(i+1)。例如 4 盤時，先以七個合法單盤移動將 D1、D2、D3
    從來源柱移至輔助柱，使 D4 暴露；再移動 D4，最後將三個小盤移到目標柱。
    本函式只保證演算法序列，
    真正的剛體位置與移動合法性由 tower.py 的物理控制流程負責。

    正確性可對 count 作歸納：0 盤時不需移動；假設 count-1 盤的子問題
    都能合法完成，第一次遞迴便會讓最大盤露出，故最大盤可合法移動；第二次遞迴
    再把小盤合法放回其上，於是 count 盤也能完成。最優性來自下界：移動最大盤
    前，小盤必須全在輔助柱；移動最大盤後，仍須把小盤移到目標柱。因此
    T(0)=0、T(n)=2T(n-1)+1。改寫為 T(n)+1=2(T(n-1)+1) 並反覆代入，
    得 T(n)+1=2**n，故任何解至少 2**n-1 步；本遞迴恰好達到此下界。

    以 3 盤為例，H(n, 來源, 目標, 暫存) 的遞迴展開如下：

        H(3, A, C, B)
        +-- H(2, A, B, C)      先把 D1、D2 移到 B
        |   +-- H(1, A, C, B)
        |   |   +-- H(0, A, B, C)  空集合，直接返回
        |   |   +-- move D1: A -> C
        |   |   +-- H(0, B, C, A)  空集合，直接返回
        |   +-- move D2: A -> B
        |   +-- H(1, C, B, A)  把 D1 放到 D2 上
        +-- move D3: A -> C   最大盤現在露出，可合法搬動
        +-- H(2, B, C, A)      最後把 D1、D2 搬到 C

    每個 H(k) 都先解一個 H(k-1)，再移動一片最大盤，最後再解另一個 H(k-1)。
    這個展開也對應程式中兩次遞迴呼叫與中間一次 moves.append()。

    Args:
        disk_count: 盤子數量，必須至少為 1。

    Returns:
        依執行順序排列的移動 tuple 清單，長度為 2**disk_count - 1。

    Raises:
        ValueError: disk_count 小於 1。
    """
    if disk_count < 1:
        raise ValueError("disk_count 至少為 1")

    moves: list[Move] = []

    def move_stack(count: int, source: int, destination: int, spare: int) -> None:
        # 契約：移動 source 頂端 count 個盤子至 destination；spare 提供唯一暫存柱。
        # 前提是來源子堆疊合法，且 destination 上的盤子都大於此子堆疊中的盤子。
        # 後置條件是這 count 個盤子依原順序位於 destination，且每一步均保持合法。
        if count == 0:
            # 基底情形：空集合已滿足移動目標，故不需產生任何操作。
            return

        # 歸納步驟一：依歸納假設，先將 count-1 個較小盤合法移至 spare，
        # 使目前最大盤 count-1 暴露於 source 頂端。
        # 子問題的柱位參數是角色而非固定名稱：此處 spare 成為目的柱，destination
        # 暫作子問題的暫存柱，才能替最大盤保留空的落點。
        move_stack(count - 1, source, spare, destination)

        # 較大盤可放在空柱或更大盤之上，因此此移動合法。
        moves.append((count - 1, source, destination))

        # 歸納步驟二：再次依歸納假設，將 spare 上的 count-1 個盤子移至
        # destination；遞迴契約遂對 count 個盤子成立。
        move_stack(count - 1, spare, destination, source)

    # 每個非基底問題由兩個 count-1 子問題及一次最大盤移動組成；遞迴式與最優性
    # 下界的推導見函式說明。這個構造達到下界，因此產生最少移動數的合法解。
    # 初始化柱位：0 為來源、2 為目標、1 為暫存。
    move_stack(disk_count, 0, 2, 1)
    return moves


def _state_neighbors(state: _State) -> list[tuple[_State, Move]]:
    """列舉從給定狀態出發的所有合法單步移動。

    對每根柱子的頂盤，嘗試移到其他兩根柱子：
    若目標柱為空，或目標柱頂盤的索引（尺寸）大於來源頂盤索引，則移動合法
    （索引越大代表盤子越大；不能把大盤疊到小盤上）。

    Args:
        state: 目前的三柱堆疊狀態（tuple of tuple，不可變）。

    Returns:
        (新狀態, Move) 配對的清單；Move 為 (盤子索引, 來源柱, 目標柱)。
    """
    neighbors: list[tuple[_State, Move]] = []
    stacks = [list(s) for s in state]  # 轉成可變副本

    for src in range(3):
        if not stacks[src]:
            continue  # 來源柱為空，無法移動
        top_disk = stacks[src][-1]

        for dst in range(3):
            if src == dst:
                continue
            # 合法條件：目標柱為空，或目標柱頂盤比當前頂盤大
            if not stacks[dst] or stacks[dst][-1] > top_disk:
                new_stacks = [list(s) for s in stacks]
                new_stacks[src].pop()
                new_stacks[dst].append(top_disk)
                new_state: _State = tuple(tuple(s) for s in new_stacks)
                neighbors.append((new_state, (top_disk, src, dst)))

    return neighbors


def plan_hanoi_from_state(
    initial_stacks: list[list[int]],
    disk_count: int,
    goal_peg: int = 2,
) -> list[Move]:
    """以 BFS 在 3^N 狀態圖中找從任意合法初始配置到目標柱的最優移動序列。

    河內塔的合法狀態空間是一個有向圖：每個節點是一個符合「大在下小在上」約束的
    三柱堆疊配置，共有 3^N 個節點（N 個盤子，每盤可在三根柱子之一）。BFS 在此圖
    中尋找從 initial_state 到 goal_state 的最短路徑，回傳最優移動序列。

    與 plan_hanoi() 的差異：plan_hanoi() 僅支援「所有盤子從柱 0 移到柱 2」的
    標準問題；本函式接受任意合法初始狀態，因此支援隨機初始堆疊。

    時間複雜度：O(3^N × N)，N≤10 約 60K 節點，遠小於毫秒等級的瓶頸。
    空間複雜度：O(3^N)，存放 visited 與 parent 字典。

    Args:
        initial_stacks: 初始三柱堆疊，格式為 [[柱0盤子從底到頂], [柱1], [柱2]]；
            盤子以 0 起算的索引表示，0 為最小盤，disk_count-1 為最大盤。
        disk_count: 盤子總數，用於建立目標狀態。
        goal_peg: 目標柱編號（預設 2，即 C 柱）。

    Returns:
        最優移動序列，格式同 plan_hanoi()：list of (disk_index, src_peg, dst_peg)。
        若初始狀態已等於目標狀態，回傳空清單。

    Raises:
        ValueError: 初始堆疊不合法（盤數不符、大盤疊在小盤上）。
        ValueError: initial_stacks 中盤子總數不等於 disk_count。
    """
    # ── 驗證輸入 ─────────────────────────────────────────────────────────────
    all_disks = [d for peg in initial_stacks for d in peg]
    if sorted(all_disks) != list(range(disk_count)):
        raise ValueError(
            f"initial_stacks 包含的盤子索引不正確：期望 0~{disk_count - 1}，"
            f"實際 {sorted(all_disks)}"
        )
    for peg_idx, peg in enumerate(initial_stacks):
        for i in range(len(peg) - 1):
            if peg[i] < peg[i + 1]:
                raise ValueError(
                    f"柱 {peg_idx} 違反約束：盤 {peg[i]} 在盤 {peg[i + 1]} 下方，但前者較小"
                )

    # ── 建立起始與目標狀態 ───────────────────────────────────────────────────
    # 目標狀態：所有盤子從大到小（底到頂）排在 goal_peg；其餘柱子為空
    goal_disks = tuple(range(disk_count - 1, -1, -1))  # 底→頂：最大盤先，最小盤後
    goal_parts: list[tuple[int, ...]] = [tuple() for _ in range(3)]
    goal_parts[goal_peg] = goal_disks
    goal_state: _State = tuple(goal_parts)  # type: ignore[arg-type]

    start_state: _State = tuple(tuple(peg) for peg in initial_stacks)  # type: ignore[arg-type]

    if start_state == goal_state:
        return []  # 已在目標，不需任何移動

    # ── BFS ──────────────────────────────────────────────────────────────────
    # parent[state] = (prev_state, move) 用於回溯路徑；起點的 parent 設為 None
    parent: dict[_State, tuple[_State, Move] | None] = {start_state: None}
    queue: deque[_State] = deque([start_state])

    while queue:
        current = queue.popleft()
        for next_state, move in _state_neighbors(current):
            if next_state in parent:
                continue
            parent[next_state] = (current, move)
            if next_state == goal_state:
                # 找到目標，回溯建立移動序列
                path: list[Move] = []
                state = next_state
                while parent[state] is not None:
                    prev, m = parent[state]
                    path.append(m)
                    state = prev
                path.reverse()
                return path
            queue.append(next_state)

    # 理論上 3 根柱子的合法狀態圖是強連通的，不應到達這裡
    raise ValueError(
        f"BFS 無法從初始狀態到達目標柱 {goal_peg}；請檢查 initial_stacks 是否合法"
    )
