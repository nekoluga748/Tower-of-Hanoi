"""把 Hanoi 符號規劃接到 Isaac Sim 剛體場景的任務控制器。

NVIDIA Franka Pick-and-Place 示範的單次抓放控制單位由 FrankaPickPlace 提供；
本模組在其上加入盤子集合、三柱符號堆疊、遞迴移動序列及每步完成／失敗判定。
每筆移動開始時，符號規劃選出來源盤與目的柱；控制器再從 USD Stage 的剛體世界位姿
取得實際放置高度，將「物件目標」換算成「末端目標」。只有一次完整取放通過後，
才更新 _stacks 和 _move_index，使離散任務狀態不會先於物理操作提交。

NVIDIA 依賴集中在 Isaac Sim 實驗性 Cube、RigidPrim、GeomPrim、材質及 Stage API；
Hanoi 遞迴、盤位狀態同步與抓取／放置檢查是本專案邏輯。
"""

from __future__ import annotations

import numpy as np
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
import isaacsim.core.experimental.utils.stage as stage_utils

from .planner import Move, plan_hanoi, plan_hanoi_from_state
from .pick_place import FrankaPickPlace


class FrankaTowerOfHanoi(FrankaPickPlace):
    """在三柱剛體場景中逐步執行 plan_hanoi() 的 Franka Hanoi 控制器。

    一筆 Hanoi 移動包含 FrankaPickPlace 的五個 Phase。_moves 是預先計算的
    符號解；_stacks 由底至頂保存各柱盤子索引，索引 0 對應最小盤。每次啟動移動時，
    本類別將來源盤設為基礎控制器目前的 cube，計算目的盤中心，再交由基礎 FSM
    下達手臂動作。抓取以盤子實際位移確認；放置以盤子世界位置校正，並允許最多兩次
    重降。只有基礎 FSM 完成且檢查未使任務失敗，符號堆疊才提交該筆移動。
    """

    def __init__(self, disk_count: int = 3, random_init: bool = False, seed: int | None = None) -> None:
        """建立 Hanoi 任務參數及初始符號堆疊。

        參數：
            disk_count: 盤子數量，支援 2 至 10 盤；移動數為 2**disk_count - 1（標準）或
                BFS 最優步數（隨機初始）。
            random_init: 為 True 時隨機打散初始堆疊，並以 BFS 規劃最優解至 C 柱。
            seed: 隨機種子；未提供時每次執行結果不同，提供時可重現相同配置。

        disk_widths 由小至大排列，索引 0 是頂層最小盤；_stacks[0] 以底盤在前、
        頂盤在後保存初始順序。實際幾何單位為公尺，物理姿態由 Isaac Sim 回報。
        """
        if not 2 <= disk_count <= 10:
            raise ValueError("disk_count 必須介於 2 到 10 之間 (最大 1023 步)")

        super().__init__(robot_name="Hanoi")
        self.disk_count = disk_count
        self._random_init = random_init
        self._rng = np.random.default_rng(seed)
        # 盤厚固定為 25 mm；增加盤數只增加堆疊高度，不縮放積木尺寸。
        self.disk_thickness = 0.025
        # 三根柱子（放置區）在世界座標系中的 (X, Y) 位置
        # 柱子間距收斂至 [-0.28, 0.0, +0.28]，遠離機械手臂全伸展奇異點
        self.peg_x = 0.50   # 柱子 X（公尺）；要再往正偏就加大、往負偏就減小（原 0.48）
        self.peg_positions = np.array([[self.peg_x, -0.28], [self.peg_x, 0.0], [self.peg_x, 0.28]])
        self.marker_size = 0.20
        self.marker_thickness = 0.001
        self.marker_colors = ["cyan", "gold", "limegreen"]

        # 盤寬差保持明顯；三盤時寬度為 3.2、4.4、5.6 cm。
        width_span = min(0.024, 0.012 * (disk_count - 1))
        self.disk_widths = np.linspace(0.056 - width_span, 0.056, disk_count)

        # 12 色高對比飽和調色盤（每層色彩分明，視覺效果優異）
        self.disk_colors = [
            "red",
            "darkorange",
            "gold",
            "yellow",
            "lawngreen",
            "limegreen",
            "darkturquoise",
            "dodgerblue",
            "blue",
            "mediumpurple",
            "darkviolet",
            "magenta",
        ]

        # 各階段運動參數配置
        self.approach_height = 0.0885       # 盤面頂部至手掌目標，避免手指跨入下層盤
        self._default_approach_height = self.approach_height
        self.retreat_height = 0.16           # 放置完畢後的垂直退避高度
        self.carry_reserve = 0.12            # 盤底高於最高堆疊的水平搬運淨空 (m)
        self.grasp_lift_threshold = 0.035    # 判定抓取成功的垂直提升閾值
        self.grasp_xy_tolerance = 0.018      # 抓取提升過程允許的水平滑移容許值（精度細化至原本 2 倍）
        self.placement_tolerance = 0.008     # 放置誤差容許值（原 38mm 比盤厚還大，半空也會通過）
        self.max_placement_corrections = 4   # 重降修正次數上限（容許值收緊後多給一次機會）
        self.pickup_probe_step = 0.005       # 抓取失敗時最多向下探 5 mm
        self.minimum_hand_target_z = 0.10    # 手掌目標離地至少 10 cm，保留夾爪與地面淨空
        self.max_pickup_attempts = 2         # 初次抓取加一次受限下探
        self.double_grasp_tolerance = 0.005  # 下層支撐盤上升超過 5 mm 視為雙抓

        # 實體剛體與狀態變數
        self.disks: list[RigidPrim] = []
        # 符號堆疊：先以標準配置初始化，setup_scene() 中若啟用隨機初始會覆寫。
        self._stacks: list[list[int]] = [list(range(disk_count - 1, -1, -1)), [], []]
        self._initial_stacks: list[list[int]] = [list(s) for s in self._stacks]
        # 移動序列：標準模式用遞迴最優解；隨機模式在 setup_scene() 覆寫為 BFS 解。
        self._moves: list[Move] = plan_hanoi(disk_count)
        self._move_index = 0
        self._pickup_attempts = 0
        self._pickup_start_z: float | None = None
        self._grasp_object_offset: np.ndarray | None = None
        self._destination_disk_position: np.ndarray | None = None
        self._support_disk_index: int | None = None
        self._support_disk_start_z: float | None = None
        self._placement_corrections = 0
        self._placement_correction_active = False
        self._failed = False

    def _disk_size(self, disk_index: int) -> np.ndarray:
        """回傳盤子的 (X 寬度, Y 寬度, Z 厚度) 尺寸，單位為公尺。"""
        width = self.disk_widths[disk_index]
        return np.array([width, width, self.disk_thickness])

    def _generate_random_stacks(self) -> list[list[int]]:
        """隨機產生一個合法的三柱初始堆疊配置，且不與目標狀態相同。

        演算法：由最大盤（索引 disk_count-1）到最小盤（索引 0）依序處理，
        每個盤子隨機選擇一根柱子放置。由於大盤先放、小盤後放，自動保證
        每根柱子內「大在下、小在上」的合法性約束。

        目標狀態（全在 C 柱）會被排除，確保至少有 1 個移動步驟需要執行。
        若盤子數 >= 2，隨機配置幾乎不可能恰好是目標狀態（概率 1/3^N），
        最多重試 100 次即可保證找到非目標配置。

        Returns:
            三根柱子的堆疊配置，格式為 [[柱0盤子從底到頂], [柱1], [柱2]]。
        """
        # 目標狀態：全在柱 2（C 柱），由大到小堆疊
        goal_stacks = [[], [], list(range(self.disk_count - 1, -1, -1))]

        for _attempt in range(100):
            stacks: list[list[int]] = [[], [], []]
            # 由最大盤到最小盤依序分配（保證大在下、小在上）
            for disk_index in range(self.disk_count - 1, -1, -1):
                peg = int(self._rng.integers(0, 3))
                stacks[peg].append(disk_index)
            if stacks != goal_stacks:
                return stacks

        # 極罕見情況（全部嘗試都恰好是目標態），fallback：把最小盤放到 A 柱
        fallback = [[], [], list(range(self.disk_count - 1, -1, -1))]
        fallback[2].pop()          # 移除最小盤（索引 0）
        fallback[0].append(0)      # 放到 A 柱
        return fallback

    def _disk_position(self, disk_index: int, peg_index: int, level: int) -> np.ndarray:
        """依柱位與堆疊層數計算盤子幾何中心的世界座標。

        level=0 表示盤子直接放在地面；更高層中心每次增加一個盤厚。
        此函式計算理想初始／目標位置；執行中的實際位置仍由 RigidPrim 回報。
        """
        peg_x, peg_y = self.peg_positions[peg_index]
        center_z = (level + 0.5) * self.disk_thickness
        return np.array([peg_x, peg_y, center_z])

    def _initial_disk_position(self, disk_index: int) -> np.ndarray:
        """依 _initial_stacks 查詢盤子的初始世界座標（幾何中心）。

        掃描三根柱子找出 disk_index 所在的位置與層數，再以 _disk_position() 換算。
        """
        for peg_index, peg in enumerate(self._initial_stacks):
            if disk_index in peg:
                level = peg.index(disk_index)  # peg 由底至頂，index 即層數
                return self._disk_position(disk_index, peg_index, level)
        raise RuntimeError(f"disk {disk_index} 不在 _initial_stacks 中")

    def setup_scene(self) -> None:
        """在目前 USD Stage 建立 Franka、盤子剛體、地面、柱框標籤及物理材質。

        若 random_init=True，先產生隨機合法初始堆疊並記錄到 _initial_stacks，
        再依堆疊決定每個盤子的初始世界座標；標準模式則全部放在 A 柱。
        透過 NVIDIA Isaac Sim 的 Cube、GeomPrim、RigidPrim 和
        RigidBodyMaterial API 建立或包裝幾何；盤子加入碰撞 API，標記僅作視覺用途。
        建立完成後載入第一筆移動，但尚未播放模擬；應用入口會啟動 Timeline 並 reset。
        """
        # ── 決定初始堆疊配置 ─────────────────────────────────────────────────
        if self._random_init:
            self._initial_stacks = self._generate_random_stacks()
            self._stacks = [list(s) for s in self._initial_stacks]
            mode_desc = "隨機初始堆疊"
        else:
            # 標準模式：所有盤子在 A 柱（由大到小堆疊）
            self._initial_stacks = [list(range(self.disk_count - 1, -1, -1)), [], []]
            self._stacks = [list(s) for s in self._initial_stacks]
            mode_desc = "標準初始堆疊（全在 A 柱）"

        # 印出初始配置讓使用者一眼確認
        peg_names = ("A", "B", "C")
        stack_desc = "  ".join(
            f"{peg_names[i]}:[{','.join(f'D{d+1}' for d in self._initial_stacks[i])}]"
            for i in range(3)
        )
        print(f"[Hanoi] {mode_desc}: {stack_desc}", flush=True)

        bottom_disk = self.disk_count - 1

        # ── 基底場景與機械手臂（以最大盤的初始位置傳給父類別）─────────────────
        super().setup_scene(
            cube_initial_position=self._initial_disk_position(bottom_disk),
            cube_size=self._disk_size(bottom_disk),
            target_position=self._disk_position(0, 2, 0) + np.array([0.0, 0.0, 0.1]),
            cube_path=f"/World/Hanoi/disks/disk_{bottom_disk}",
        )

        self.disks = [None] * self.disk_count
        self.disks[bottom_disk] = self.cube

        # ── 其餘盤子依初始堆疊位置建立 ───────────────────────────────────────
        for disk_index in range(bottom_disk):
            disk_path = f"/World/Hanoi/disks/disk_{disk_index}"
            Cube(
                paths=disk_path,
                positions=self._initial_disk_position(disk_index),
                sizes=1.0,
                scales=self._disk_size(disk_index),
                colors=self.disk_colors[disk_index % len(self.disk_colors)],
            )
            GeomPrim(paths=disk_path, apply_collision_apis=True)
            self.disks[disk_index] = RigidPrim(paths=disk_path)

        # 柱框與 A/B/C 字形由非剛體 Cube 構成，不加入碰撞，避免視覺標記干擾抓放。
        label_strokes = {
            "A": [((-0.5, 0.0), (0.0, 1.0)), ((0.0, 1.0), (0.5, 0.0)), ((-0.25, 0.42), (0.25, 0.42))],
            "B": [((-0.5, 0.0), (-0.5, 1.0)), ((-0.5, 1.0), (0.2, 1.0)), ((-0.5, 0.5), (0.2, 0.5)), ((-0.5, 0.0), (0.2, 0.0)), ((0.2, 1.0), (0.45, 0.8)), ((0.45, 0.8), (0.45, 0.65)), ((0.45, 0.65), (0.2, 0.5)), ((0.2, 0.5), (0.45, 0.35)), ((0.45, 0.35), (0.45, 0.2)), ((0.45, 0.2), (0.2, 0.0))],
            "C": [((-0.5, 0.0), (-0.5, 1.0)), ((-0.5, 1.0), (0.5, 1.0)), ((-0.5, 0.0), (0.5, 0.0))],
        }
        label_width = 0.05
        label_height = 0.07
        label_stroke = 0.001
        for peg_index, (peg_x, peg_y) in enumerate(self.peg_positions):
            marker_color = self.marker_colors[peg_index]
            marker_z = self.marker_thickness / 2
            edge_offset = self.marker_size / 2 - self.marker_thickness / 2
            marker_edges = [
                ("north", [peg_x, peg_y + edge_offset, marker_z], [self.marker_size, self.marker_thickness, self.marker_thickness]),
                ("south", [peg_x, peg_y - edge_offset, marker_z], [self.marker_size, self.marker_thickness, self.marker_thickness]),
                ("east", [peg_x + edge_offset, peg_y, marker_z], [self.marker_thickness, self.marker_size, self.marker_thickness]),
                ("west", [peg_x - edge_offset, peg_y, marker_z], [self.marker_thickness, self.marker_size, self.marker_thickness]),
            ]
            for edge_name, position, scale in marker_edges:
                Cube(
                    paths=f"/World/Hanoi/markers/zone_{peg_index}_{edge_name}",
                    positions=np.array(position),
                    sizes=1.0,
                    scales=np.array(scale),
                    colors=marker_color,
                )

            label_center_x = peg_x - self.marker_size / 2 - label_width / 2 - 0.035
            label_center_y = peg_y + label_height / 2
            for stroke_index, (start, end) in enumerate(label_strokes["ABC"[peg_index]]):
                start_x = label_center_x + (start[1] - 0.5) * label_height
                start_y = label_center_y - (start[0] * label_width)
                end_x = label_center_x + (end[1] - 0.5) * label_height
                end_y = label_center_y - (end[0] * label_width)
                delta_x = end_x - start_x
                delta_y = end_y - start_y
                stroke_length = float(np.hypot(delta_x, delta_y))
                angle = float(np.arctan2(delta_y, delta_x))
                Cube(
                    paths=f"/World/Hanoi/markers/peg_label_{peg_index}_{stroke_index}",
                    positions=np.array([(start_x + end_x) / 2, (start_y + end_y) / 2, label_stroke / 2]),
                    orientations=np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)]),
                    sizes=1.0,
                    scales=np.array([stroke_length, label_stroke, label_stroke]),
                    colors="black",
                )

        # NVIDIA RigidBodyMaterial 套用至盤子幾何，影響 PhysX 接觸摩擦與回彈，不改變符號解。
        disk_geometry = GeomPrim(paths=[f"/World/Hanoi/disks/disk_{index}" for index in range(self.disk_count)])
        disk_material = RigidBodyMaterial(
            "/World/PhysicsMaterials/hanoi_disks",
            static_frictions=2.0,
            dynamic_frictions=1.5,
            restitutions=0.0,
        )
        disk_geometry.apply_physics_materials(disk_material)

        # ── 依初始堆疊產生移動規劃序列，並啟動第一步 ────────────────────────
        if self._random_init:
            self._moves = plan_hanoi_from_state(self._initial_stacks, self.disk_count)
            print(f"[Hanoi] BFS 規劃完成，共需 {len(self._moves)} 步。", flush=True)
        else:
            self._moves = plan_hanoi(self.disk_count)

        self._activate_current_move()

    def _safe_transit_z(self) -> float:
        """依目前最高符號堆疊估算水平搬運與撤離時的末端巡航安全 Z 高度。

        地面碰撞面位於 z=0，因此最高堆疊頂面為層數乘以盤厚。
        再加盤厚、夾爪至盤面的間距及 carry_reserve，確保盤底保有指定淨空；結果
        限制在 0.34–0.52 m 的可用工作高度範圍內。
        """
        tallest_stack = max(len(stack) for stack in self._stacks)
        tallest_stack_top = tallest_stack * self.disk_thickness
        computed = (
            tallest_stack_top
            + self.disk_thickness
            + self.approach_height
            + self.carry_reserve
        )
        return float(np.clip(computed, 0.34, 0.52))

    def _activate_current_move(self) -> None:
        """把目前符號移動轉成基礎 Pick-and-Place 控制器的物件與目標位姿。

        X/Y 目標固定在目的柱中心，以抑制堆疊誤差累積；若目的柱已有盤子，Z 目標
        取其最上層剛體的實測中心加一個盤厚；水平搬運高度由 _safe_transit_z() 統一計算。
        """
        disk_index, _, destination = self._moves[self._move_index]
        destination_level = len(self._stacks[destination])

        self.cube = self.disks[disk_index]
        peg_x, peg_y = self.peg_positions[destination]

        # 核心防飄移修正：X, Y 永遠對準柱子正中心 (peg_x, peg_y)，防止誤差累積連鎖歪斜
        # Z 座標則精準貼合目標柱上既有方塊的實際物理頂部
        if destination_level > 0:
            below_disk_idx = self._stacks[destination][-1]
            below_pos = self.disks[below_disk_idx].get_world_poses()[0].numpy()[0]
            self._destination_disk_position = np.array(
                [peg_x, peg_y, below_pos[2] + self.disk_thickness]
            )
        else:
            self._destination_disk_position = self._disk_position(disk_index, destination, 0)

        self.target_position = self._destination_disk_position.copy()
        self.approach_height = self._default_approach_height
        self._pickup_attempts = 0
        self._pickup_start_z = None
        self._grasp_object_offset = None
        self._placement_corrections = 0
        self._placement_correction_active = False
        self._support_disk_index = None
        self._support_disk_start_z = None

    def _retry_failed_pickup(self) -> bool:
        """未確認盤子被抬起時，回到安全高度並進行一次受限下探。

        抬升不足或水平漂移超過閾值時呼叫。最多向下探一次 pickup_probe_step，且
        手掌世界 Z 目標不得低於 minimum_hand_target_z；若沒有安全空間或探測仍失敗，
        設定 _failed 停止後續移動。
        回傳值表示是否允許繼續任務。
        """
        self._pickup_attempts += 1
        self.robot.open_gripper()

        if self._pickup_attempts >= self.max_pickup_attempts:
            self._failed = True
            print("Hanoi stopped: disk was not lifted after one safe downward probe.")
            return False

        disk_position = self.cube.get_world_poses()[0].numpy()[0]
        minimum_approach_height = max(
            0.0,
            self.minimum_hand_target_z
            - float(disk_position[2])
            - self.disk_thickness / 2,
        )
        next_approach_height = max(
            minimum_approach_height,
            self.approach_height - self.pickup_probe_step,
        )
        probe_distance = self.approach_height - next_approach_height
        if probe_distance <= 1e-6:
            self._failed = True
            print(
                "Hanoi stopped: no safe downward probe remains; "
                f"hand target must stay at or above {self.minimum_hand_target_z:.3f} m."
            )
            return False

        self.approach_height = next_approach_height
        self._event = 0
        self._step = 0
        self._phase_start_position = None
        self._support_disk_index = None
        self._support_disk_start_z = None
        self._pickup_start_z = None
        self._grasp_object_offset = None
        print(
            f"Grasp not confirmed; returning to clearance and probing down "
            f"{probe_distance:.3f} m once with {self.approach_height:.3f} m clearance above the disk."
        )
        return True

    @staticmethod
    def _hand_target_for_disk(disk_target: np.ndarray, disk_hand_offset: np.ndarray) -> np.ndarray:
        return disk_target - disk_hand_offset

    def _support_disk_was_lifted(self, lift_distance: float) -> bool:
        return lift_distance > self.double_grasp_tolerance

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """執行一影格 Hanoi 任務控制，將符號移動和物理抓放串接。

        呼叫端需在模擬正在播放時逐影格呼叫。每影格先處理基於實測剛體位姿的抓取／
        放置檢查，再呼叫基礎 FSM 下達末端目標。基礎 FSM 全部完成後才更新來源和目的
        _stacks，並啟動下一筆移動。回傳值沿用基礎控制器的執行狀態；任務最終狀態
        請以 is_done() 判斷。
        """
        if self._failed or self._move_index >= len(self._moves):
            return False
        if self._motion_failed:
            self._failed = True
            print("Hanoi stopped: arm did not reach a required pose before timeout.")
            return False

        if self._event == 1 and self._step == 0 and self._support_disk_index is None:
            _, source, _ = self._moves[self._move_index]
            if len(self._stacks[source]) > 1:
                self._support_disk_index = self._stacks[source][-2]
                support_position = self.disks[self._support_disk_index].get_world_poses()[0].numpy()[0]
                self._support_disk_start_z = float(support_position[2])

        # ── 偵測一次完整 Pick-Place (Phase 0~4) 是否執行完畢 ─────────────────
        if super().is_done():
            if self._destination_disk_position is None:
                self._failed = True
                print("Hanoi stopped: destination pose was not initialized.")
                return False

            disk_position = self.cube.get_world_poses()[0].numpy()[0]
            settled_error = float(
                np.linalg.norm(self._destination_disk_position - disk_position)
            )
            if settled_error > self.placement_tolerance:
                self._failed = True
                gripper_positions = self.robot.get_dof_positions().numpy()[0, 7:9]
                print(
                    f"Hanoi stopped after release: disk settled {settled_error:.3f} m "
                    f"off target; target={self._destination_disk_position.tolist()}, "
                    f"disk={disk_position.tolist()}, finger positions={gripper_positions.tolist()}; "
                    "move was not committed."
                )
                return False

            # 只在完整抓放 FSM 結束後提交符號移動；前面的階段不提前改寫柱位狀態。
            disk_index, source, destination = self._moves[self._move_index]
            moved_disk = self._stacks[source].pop()
            if moved_disk != disk_index:
                raise RuntimeError("Hanoi move plan no longer matches the simulated stacks")
            self._stacks[destination].append(disk_index)
            self._move_index += 1

            # 判斷是否所有步數皆完成
            if self._move_index >= len(self._moves):
                return False

            # 切換至下一步：重置狀態機並載入下一步動作
            self._event = 0
            self._step  = 0
            self._phase_start_position = None
            self._activate_current_move()

        # 在 Phase 1 下潛段轉入底部等待時記錄盤子與手部相對位姿，用於修正末端目標。
        seg = max(self.grasp_descent_steps, int(self.events_dt[1]) // 3)
        if self._event == 1 and self._step == seg:
            # 記錄方塊相對夾爪偏移，以修正抓取後搬運的末端目標
            disk_position = self.cube.get_world_poses()[0].numpy()
            _, hand_position, _   = self.robot.get_current_state()
            self._grasp_object_offset = disk_position[0] - hand_position[0]
            # 目標位置修正：手臂需移至 (目標盤子位置 - 夾爪相對偏移)
            self.target_position = (
                self._destination_disk_position - self._grasp_object_offset
            )

        # 閉爪會使盤子在指間重新置中；抓取穩定度基準應取在閉爪後、抬升前。
        if (
            self._event == 1
            and self._grasp_close_start_step is not None
            and self._step
            == self._grasp_close_start_step
            + max(self._scale_steps(12), self.grasp_close_steps)
        ):
            disk_position = self.cube.get_world_poses()[0].numpy()
            _, hand_position, _ = self.robot.get_current_state()
            self._pickup_start_z = float(disk_position[0, 2])
            self._grasp_object_offset = disk_position[0] - hand_position[0]

        # ── Phase 2 開始時：檢查方塊是否確實被夾起（防抓空氣）──────────────
        if self._event == 2 and self._step == 0 and self._pickup_start_z is not None:
            if self._support_disk_index is not None and self._support_disk_start_z is not None:
                support_position = self.disks[self._support_disk_index].get_world_poses()[0].numpy()[0]
                support_lift = float(support_position[2]) - self._support_disk_start_z
                if self._support_disk_was_lifted(support_lift):
                    self._failed = True
                    print(
                        f"Double grasp detected: supporting disk D{self._support_disk_index + 1} "
                        f"lifted {support_lift:.3f} m; stopping before carry."
                    )
                    return False

            disk_position  = self.cube.get_world_poses()[0].numpy()
            lifted_distance = float(disk_position[0, 2]) - self._pickup_start_z
            _, hand_position, _ = self.robot.get_current_state()
            if self._grasp_object_offset is None:
                raise RuntimeError("Pickup pose was not recorded before the grasp check")
            current_grasp_offset = disk_position[0] - hand_position[0]
            relative_xy_drift = float(
                np.linalg.norm(current_grasp_offset[:2] - self._grasp_object_offset[:2])
            )
            disk_index = self._moves[self._move_index][0]
            allowed_xy_drift = max(
                self.grasp_xy_tolerance,
                float(self.disk_widths[disk_index]) * 1.25,
            )
            if lifted_distance < self.grasp_lift_threshold or relative_xy_drift > allowed_xy_drift:
                print(
                    f"Pickup check failed: lift={lifted_distance:.3f} m, "
                    f"relative horizontal drift={relative_xy_drift:.3f} m "
                    f"(limit={allowed_xy_drift:.3f} m)."
                )
                return self._retry_failed_pickup()
            # 以抬升後的實測物件－手部偏移換算末端目標，避免假設夾爪正好抓在盤子中心。
            self._grasp_object_offset = current_grasp_offset
            self.target_position = self._hand_target_for_disk(
                self._destination_disk_position,
                self._grasp_object_offset,
            )
            self._pickup_start_z = None

        # ── Phase 3 開始時：重新校正目標位置（防止長途搬運誤差累積）─────────
        if (
            self._event == 3
            and self._step == 0
            and self._destination_disk_position is not None
            and not self._placement_correction_active
        ):
            _, _, destination = self._moves[self._move_index]
            destination_level = len(self._stacks[destination])
            peg_x, peg_y = self.peg_positions[destination]
            # 若目標柱已有盤子，以實際物理位置計算精確疊放高度
            if destination_level > 0:
                below_disk_idx = self._stacks[destination][-1]
                below_pos = self.disks[below_disk_idx].get_world_poses()[0].numpy()[0]
                self._destination_disk_position = np.array(
                    [peg_x, peg_y, below_pos[2] + self.disk_thickness]
                )
            disk_position = self.cube.get_world_poses()[0].numpy()[0]
            _, hand_position, _ = self.robot.get_current_state()
            self._grasp_object_offset = disk_position - hand_position[0]
            self.target_position = self._hand_target_for_disk(
                self._destination_disk_position,
                self._grasp_object_offset,
            )

        # ── Phase 4 開始時：閉迴路檢查對準精度（若偏移過大則重降一次修正）──
        if self._event == 4 and self._step == 0 and self._destination_disk_position is not None:
            disk_position   = self.cube.get_world_poses()[0].numpy()[0]
            placement_delta = self._destination_disk_position - disk_position
            placement_error = float(np.linalg.norm(placement_delta))
            if placement_error > self.placement_tolerance:
                if self._placement_corrections >= self.max_placement_corrections:
                    self._failed = True
                    print(
                        f"Hanoi stopped before release: disk remains "
                        f"{placement_error:.3f} m off target after {self.max_placement_corrections} corrections."
                    )
                    return False
                # 退回 Phase 3 重新對準
                _, hand_position, _ = self.robot.get_current_state()
                self.target_position = (
                    hand_position[0]
                    + placement_delta
                )
                self._placement_corrections += 1
                self._event = 3
                self._step  = 0
                self._phase_start_position = None
                self._placement_correction_active = True
                # print(
                #     f"Placement correction {self._placement_corrections}/{self.max_placement_corrections}: "
                #     f"error=({placement_delta[0]:.3f}, {placement_delta[1]:.3f}, "
                #     f"{placement_delta[2]:.3f}) m; retrying descent."
                # )
                return True
            self._placement_correction_active = False

        # ── 執行基礎 FSM 一步進 ───────────────────────────────────────────────
        step_executed = super().forward(ik_method)
        if self._motion_failed:
            self._failed = True
            print("Hanoi stopped: arm did not reach a required pose before timeout.")
            return False

        return step_executed

    def is_done(self) -> bool:
        """回傳任務是否已到終態：所有移動完成，或失敗旗標已設定。"""
        return self._failed or self._motion_failed or self._move_index >= len(self._moves)

    def reset(self) -> None:
        """重置機械臂、盤子世界位姿、符號堆疊與移動／重試計數。

        本方法重設現有 USD/Articulation 狀態，不重新建立 Stage 幾何；適用於同一場景重跑。
        隨機初始模式下，盤子會還原到 setup_scene() 時產生的隨機配置，而非標準 A 柱配置。
        """
        self.reset_robot()
        # 還原到初始堆疊（標準模式 = 全在 A；隨機模式 = 當次隨機配置）
        self._stacks = [list(s) for s in self._initial_stacks]
        self._move_index = 0
        self._pickup_attempts = 0
        self._pickup_start_z = None
        self._grasp_object_offset = None
        self._placement_corrections = 0
        self._placement_correction_active = False
        self._failed = False

        # 依初始堆疊還原每個盤子的世界位姿
        for disk_index in range(self.disk_count):
            self.disks[disk_index].set_world_poses(
                positions=self._initial_disk_position(disk_index)
            )

        if self._moves:
            self._activate_current_move()
