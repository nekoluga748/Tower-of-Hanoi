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

from .planner import Move, plan_hanoi, required_lift_height
from .pick_place import FrankaPickPlace


class FrankaTowerOfHanoi(FrankaPickPlace):
    """在三柱剛體場景中逐步執行 plan_hanoi() 的 Franka Hanoi 控制器。

    一筆 Hanoi 移動包含 FrankaPickPlace 的五個 Phase。_moves 是預先計算的
    符號解；_stacks 由底至頂保存各柱盤子索引，索引 0 對應最小盤。每次啟動移動時，
    本類別將來源盤設為基礎控制器目前的 cube，計算目的盤中心，再交由基礎 FSM
    下達手臂動作。抓取以盤子實際位移確認；放置以盤子世界位置校正，並允許最多兩次
    重降。只有基礎 FSM 完成且檢查未使任務失敗，符號堆疊才提交該筆移動。
    """

    def __init__(self, disk_count: int = 3) -> None:
        """建立 Hanoi 任務參數及初始符號堆疊。

        參數：
            disk_count: 盤子數量，支援 2 至 10 盤；移動數為 2**disk_count - 1。

        disk_widths 由小至大排列，索引 0 是頂層最小盤；_stacks[0] 以底盤在前、
        頂盤在後保存初始順序。實際幾何單位為公尺，物理姿態由 Isaac Sim 回報。
        """
        if not 2 <= disk_count <= 10:
            raise ValueError("disk_count 必須介於 2 到 10 之間 (最大 1023 步)")

        super().__init__(robot_name="Hanoi")
        self.disk_count = disk_count
        # 盤厚隨盤數調整並限制範圍；尺寸以公尺表示，幾何中心高度由 _disk_position 計算。
        self.disk_thickness = float(np.clip(0.12 / disk_count, 0.012, 0.025))
        # 三根柱子（放置區）在世界座標系中的 (X, Y) 位置
        # 柱子間距收斂至 [-0.28, 0.0, +0.28]，遠離機械手臂全伸展奇異點
        self.peg_positions = np.array([[0.48, -0.28], [0.48, 0.0], [0.48, 0.28]])
        self.marker_size = 0.20
        self.marker_thickness = 0.001
        self.marker_colors = ["cyan", "gold", "limegreen"]

        # 盤寬差保持明顯；三盤時寬度為 3.2、4.4、5.6 cm。
        width_span = min(0.032, 0.012 * (disk_count - 1))
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
        self.pregrasp_height = 0.20          # 移動至上方時的初始離地高度
        self.approach_height = 0.106         # 抓取點較原設定提高 1 cm
        self._default_approach_height = self.approach_height
        self.lift_height = 0.14              # 抓起後的提升高度
        self.retreat_height = 0.16           # 放置完畢後的垂直退避高度
        self.carry_reserve = 0.06            # 水平巡航時高於最高塔的安全裕度
        self.grasp_lift_threshold = 0.035    # 判定抓取成功的垂直提升閾值
        self.grasp_xy_tolerance = 0.035      # 抓取提升過程允許的水平滑移容許值
        self.carry_tracking_tolerance = 0.05 # 水平搬運過程方塊相對夾爪的滑移容許值
        self.placement_tolerance = 0.075     # 最終放置誤差容許值
        self.max_pickup_attempts = 3         # 抓取失敗最大重試次數

        # 實體剛體與狀態變數
        self.disks: list[RigidPrim] = []
        self._moves: list[Move] = plan_hanoi(disk_count)
        self._move_index = 0
        # 符號堆疊由底至頂存盤子索引；起始時所有盤子都在 Peg 0。
        self._stacks: list[list[int]] = [list(range(disk_count - 1, -1, -1)), [], []]
        self._pickup_attempts = 0
        self._pickup_start_z: float | None = None
        self._pickup_start_xy: np.ndarray | None = None
        self._grasp_object_offset: np.ndarray | None = None
        self._destination_disk_position: np.ndarray | None = None
        self._placement_corrections = 0
        self._placement_correction_active = False
        self._failed = False

    def _disk_size(self, disk_index: int) -> np.ndarray:
        """回傳盤子的 (X 寬度, Y 寬度, Z 厚度) 尺寸，單位為公尺。"""
        width = self.disk_widths[disk_index]
        return np.array([width, width, self.disk_thickness])

    def _disk_position(self, disk_index: int, peg_index: int, level: int) -> np.ndarray:
        """依柱位與堆疊層數計算盤子幾何中心的世界座標。

        level=0 表示盤子直接放在地面；更高層中心每次增加一個盤厚。
        此函式計算理想初始／目標位置；執行中的實際位置仍由 RigidPrim 回報。
        """
        peg_x, peg_y = self.peg_positions[peg_index]
        center_z = (level + 0.5) * self.disk_thickness
        return np.array([peg_x, peg_y, center_z])

    def setup_scene(self) -> None:
        """在目前 USD Stage 建立 Franka、盤子剛體、地面、柱框標籤及物理材質。

        透過 NVIDIA Isaac Sim 的 Cube、GeomPrim、RigidPrim 和
        RigidBodyMaterial API 建立或包裝幾何；盤子加入碰撞 API，標記僅作視覺用途。
        建立完成後載入第一筆遞迴移動，但尚未播放模擬；應用入口會啟動 Timeline 並 reset。
        """
        bottom_disk = self.disk_count - 1
        # 初始化基底場景與機械手臂
        super().setup_scene(
            cube_initial_position=self._disk_position(bottom_disk, 0, 0),
            cube_size=self._disk_size(bottom_disk),
            target_position=self._disk_position(0, 2, 0) + np.array([0.0, 0.0, 0.1]),
            cube_path=f"/World/Hanoi/disks/disk_{bottom_disk}",
        )

        self.disks = [None] * self.disk_count
        self.disks[bottom_disk] = self.cube

        # 基礎控制器先建立最底盤；其餘盤子按索引遞增建立，世界 Z 由底往上排列。
        for disk_index in range(bottom_disk):
            disk_path = f"/World/Hanoi/disks/disk_{disk_index}"
            Cube(
                paths=disk_path,
                positions=self._disk_position(disk_index, 0, bottom_disk - disk_index),
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

            label_center_x = peg_x - self.marker_size / 2 - label_width / 2 - 0.015
            for stroke_index, (start, end) in enumerate(label_strokes["ABC"[peg_index]]):
                start_x = label_center_x + start[0] * label_width
                start_y = peg_y + start[1] * label_height
                end_x = label_center_x + end[0] * label_width
                end_y = peg_y + end[1] * label_height
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

        # 啟動第一個搬運步驟
        self._activate_current_move()

    def _safe_transit_z(self) -> float:
        """依目前最高符號堆疊估算水平搬運時的末端巡航 Z。

        估算式以最高堆疊頂面、抓取高度、半盤厚及 4 cm 裕度相加，並限制在
        0.24–0.34 m。這是末端走廊的幾何啟發式，不等於完整碰撞規劃；實際安全性
        仍取決於盤子／夾爪尺寸、當前姿態及 PhysX 場景。
        """
        tallest_stack = max(len(stack) for stack in self._stacks)
        tallest_stack_top = tallest_stack * self.disk_thickness
        margin = 0.04
        computed = tallest_stack_top + self.approach_height + self.disk_thickness / 2 + margin
        return float(np.clip(computed, 0.24, 0.34))

    def _activate_current_move(self) -> None:
        """把目前符號移動轉成基礎 Pick-and-Place 控制器的物件與目標位姿。

        X/Y 目標固定在目的柱中心，以抑制堆疊誤差累積；若目的柱已有盤子，Z 目標
        取其最上層剛體的實測中心加一個盤厚。lift_height 則由最高符號堆疊與
        required_lift_height() 計算，並再檢查是否至少能到達安全走廊。
        """
        disk_index, source, destination = self._moves[self._move_index]
        source_level = len(self._stacks[source]) - 1
        destination_level = len(self._stacks[destination])
        source_center_z = (source_level + 0.5) * self.disk_thickness
        tallest_stack = max(len(stack) for stack in self._stacks)

        # 計算越過最高塔所需的垂直提升高度
        self.lift_height = required_lift_height(
            current_disk_center_z=source_center_z,
            stack_layers=tallest_stack,
            block_height=self.disk_thickness,
            reserve=self.carry_reserve,
            minimum=0.10,
        )
        safe_z = self._safe_transit_z()
        min_lift_for_safe = safe_z - source_center_z
        if self.lift_height < min_lift_for_safe:
            self.lift_height = min_lift_for_safe

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
        self._pickup_start_xy = None
        self._grasp_object_offset = None
        self._placement_corrections = 0
        self._placement_correction_active = False

    def _retry_failed_pickup(self) -> bool:
        """未確認盤子被抬起時，重設 FSM 並以較低抓取目標重試。

        抬升不足或水平漂移超過閾值時呼叫。重試會重新張爪、清除取物基準並把
        approach_height 降低 25 mm；達到最大嘗試次數則設定 _failed，停止後續移動。
        回傳值表示是否允許繼續任務。
        """
        self._pickup_attempts += 1
        self.robot.open_gripper()

        if self._pickup_attempts >= self.max_pickup_attempts:
            self._failed = True
            print("Hanoi stopped: disk was not lifted after three grasp attempts.")
            return False

        # 下次重試時稍微下壓 2.5cm，加深抓取深度
        self.approach_height = max(
            0.035,
            self._default_approach_height - 0.025 * self._pickup_attempts,
        )
        self._event = 0
        self._step = 0
        self._phase_start_position = None
        self._pickup_start_z = None
        self._pickup_start_xy = None
        self._grasp_object_offset = None
        print(
            f"Grasp not confirmed; retrying at {self.approach_height:.3f} m approach height "
            f"({self._pickup_attempts + 1}/{self.max_pickup_attempts})."
        )
        return True

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """執行一影格 Hanoi 任務控制，將符號移動和物理抓放串接。

        呼叫端需在模擬正在播放時逐影格呼叫。每影格先處理基於實測剛體位姿的抓取／
        放置檢查，再呼叫基礎 FSM 下達末端目標。基礎 FSM 全部完成後才更新來源和目的
        _stacks，並啟動下一筆移動。回傳值沿用基礎控制器的執行狀態；任務最終狀態
        請以 is_done() 判斷。
        """
        if self._failed or self._move_index >= len(self._moves):
            return False

        # ── 偵測一次完整 Pick-Place (Phase 0~4) 是否執行完畢 ─────────────────
        if super().is_done():
            if self._destination_disk_position is None:
                self._failed = True
                print("Hanoi stopped: destination pose was not initialized.")
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
                print("Tower of Hanoi completed!")
                return False

            # 切換至下一步：重置狀態機並載入下一步動作
            self._event = 0
            self._step  = 0
            self._phase_start_position = None
            self._activate_current_move()

        # 在 Phase 1 下潛段轉入底部等待時記錄盤子與手部相對位姿，作為後續抬升檢查基準。
        seg = max(3, int(self.events_dt[1])) // 3
        if self._event == 1 and self._step == seg:
            # 記錄方塊在 V 形底部的位置與相對夾爪偏移
            disk_position = self.cube.get_world_poses()[0].numpy()
            self._pickup_start_z  = float(disk_position[0, 2])
            self._pickup_start_xy = disk_position[0, :2].copy()
            _, hand_position, _   = self.robot.get_current_state()
            self._grasp_object_offset = disk_position[0] - hand_position[0]
            # 目標位置修正：手臂需移至 (目標盤子位置 - 夾爪相對偏移)
            self.target_position = (
                self._destination_disk_position - self._grasp_object_offset
            )

        # ── Phase 2 開始時：檢查方塊是否確實被夾起（防抓空氣）──────────────
        if self._event == 2 and self._step == 0 and self._pickup_start_z is not None:
            disk_position  = self.cube.get_world_poses()[0].numpy()
            lifted_distance = float(disk_position[0, 2]) - self._pickup_start_z
            xy_drift        = float(np.linalg.norm(disk_position[0, :2] - self._pickup_start_xy))
            if lifted_distance < self.grasp_lift_threshold or xy_drift > self.grasp_xy_tolerance:
                print(
                    f"Pickup check failed: lift={lifted_distance:.3f} m, "
                    f"horizontal drift={xy_drift:.3f} m."
                )
                return self._retry_failed_pickup()
            # 以抬升後的實測物件－手部偏移換算末端目標，避免假設夾爪正好抓在盤子中心。
            _, hand_position, _ = self.robot.get_current_state()
            self._grasp_object_offset = disk_position[0] - hand_position[0]
            touchdown_offset = np.array([0.0, 0.0, -0.006])
            self.target_position = (
                self._destination_disk_position
                - self._grasp_object_offset
                + touchdown_offset
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
            # -6mm 貼地微壓：抵消 PD 控制器重力下垂穩態誤差，確保方塊緊貼面才開爪
            touchdown_offset = np.array([0.0, 0.0, -0.006])
            self.target_position = (
                self._destination_disk_position
                - self._grasp_object_offset
                + touchdown_offset
            )

        # ── Phase 4 開始時：閉迴路檢查對準精度（若偏移過大則重降一次修正）──
        if self._event == 4 and self._step == 0 and self._destination_disk_position is not None:
            disk_position   = self.cube.get_world_poses()[0].numpy()[0]
            placement_delta = self._destination_disk_position - disk_position
            placement_error = float(np.linalg.norm(placement_delta))
            if placement_error > self.placement_tolerance:
                if self._placement_corrections >= 2:
                    self._failed = True
                    print(
                        f"Hanoi stopped before release: disk remains "
                        f"{placement_error:.3f} m off target after two corrections."
                    )
                    return False
                # 退回 Phase 3 重新對準
                _, hand_position, _ = self.robot.get_current_state()
                self.target_position = hand_position[0] + placement_delta
                self._placement_corrections += 1
                self._event = 3
                self._step  = 0
                self._phase_start_position = None
                self._placement_correction_active = True
                print(
                    f"Placement correction {self._placement_corrections}/2: "
                    f"error={placement_error:.3f} m; retrying descent."
                )
                return True
            self._placement_correction_active = False

        # ── 執行基礎 FSM 一步進 ───────────────────────────────────────────────
        step_executed = super().forward(ik_method)

        return step_executed

    def is_done(self) -> bool:
        """回傳任務是否已到終態：所有移動完成，或失敗旗標已設定。"""
        return self._failed or self._move_index >= len(self._moves)

    def reset(self) -> None:
        """重置機械臂、盤子世界位姿、符號堆疊與移動／重試計數。

        本方法重設現有 USD/Articulation 狀態，不重新建立 Stage 幾何；適用於同一場景重跑。
        """
        self.reset_robot()
        self._stacks = [list(range(self.disk_count - 1, -1, -1)), [], []]
        self._move_index = 0
        self._pickup_attempts = 0
        self._pickup_start_z = None
        self._pickup_start_xy = None
        self._grasp_object_offset = None
        self._placement_corrections = 0
        self._placement_correction_active = False
        self._failed = False

        for disk_index, disk in enumerate(self.disks):
            disk.set_world_poses(positions=self._disk_position(disk_index, 0, self.disk_count - 1 - disk_index))

        if self._moves:
            self._activate_current_move()
