# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Franka 取放 (Pick-and-Place) 5 階段有限狀態機控制器。

================================================================================
機器人學課題核心概念：有限狀態機 (Finite State Machine, FSM) 動作規劃
================================================================================

本檔保留 NVIDIA Copyright 與 Apache-2.0 授權標頭，提供由 Franka 抓取／放置範例
延伸的單物件操作骨架。本專案將它整理成可重複呼叫的 FSM；tower.py 再將每一筆
Hanoi 移動映射成一次完整的五階段取放。NVIDIA Isaac Sim 負責場景、物理與 Articulation
API；此檔負責本專案的階段時序與目標位姿插值。

連續末端運動被分割為離散 Phase，每個模擬影格只產生當前階段的一個目標，之後透過
Franka.set_end_effector_pose() 依實際關節／末端狀態重新計算關節目標。

┌─────────────────────────────────────────────────────────────────────────────┐
│          一次 Hanoi 搬盤動作（1 Hanoi Step）= Phase 0 ~ 4 合計              │
├────────┬───────────────────────────────────────────────────────────────────┤
│ Phase  │ 說明                                                               │
├────────┼───────────────────────────────────────────────────────────────────┤
│   0    │ 橫向巡航：手臂移至目標方塊正上方的安全巡航高度（夾爪全開）          │
│   1    │ V 形抓取＋退避：張爪下潛、底部等末端收斂、閉爪保持、提升           │
│   2    │ 水平搬運：高空走廊橫移至目標柱正上方（S 曲線）                     │
│   3    │ 垂直緩降：降落至目標疊放位置，誤差 ≤ 10mm 才切換（杜絕半空釋放）  │
│   4    │ 開爪釋放＋退避：前段靜止開爪，後段 S 曲線退避（消除放下後擺動）    │
└────────┴───────────────────────────────────────────────────────────────────┘

位置插值採用三次 Smoothstep 公式：
    s(t) = 3t² − 2t³,  t ∈ [0, 1]
    s'(0) = 0（起始速度為零，無突跳）
    s'(1) = 0（終止速度為零）
    → 三次多項式端點加速度不為零；實際關節反應仍由 Isaac Sim 物理控制器決定。
"""

from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim

from .franka import Franka


class FrankaPickPlace:
    """Franka 模組化取放任務控制器（5 階段 FSM）。

    一個完整任務依序執行：巡航至來源上方、抓取並抬升、水平搬運、下降放置、
    張爪並退避。此類別只管理目前 cube 與 target_position；多盤來源選擇、
    三柱堆疊和合法移動由子類別 FrankaTowerOfHanoi 負責。

    Args:
        events_dt: 五個 Phase 的最大模擬影格數；None 使用內建設定。
        robot_name: 用於錯誤訊息的機器人名稱。
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 初始化
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        events_dt: list[float] | None = None,
        robot_name: str = "",
    ) -> None:
        self.cube = None
        self.robot = None
        self.robot_name = robot_name

        # ── 5 階段有限狀態機時長設定（單位：模擬步數 @ 60Hz）────────────────
        # 所有動作與等待步數均按 1.5 倍播放速度縮短，位置收斂門檻維持原值。
        # Nominal Phase durations in simulation frames, scaled uniformly below.
        self.playback_speed = 1.5
        phase_durations = events_dt if events_dt is not None else [30, 60, 48, 68, 38]
        self.events_dt = [self._scale_steps(duration) for duration in phase_durations]

        # ── 運動幾何參數 ───────────────────────────────────────────────────────
        self.pregrasp_height = 0.22   # 橫向巡航時末端執行器距方塊中心的垂直距離 (m)
        self.approach_height = 0.096  # V 形底部：末端執行器與方塊頂面的距離 (m)
        self.lift_height     = 0.22   # 提升目標高度，蓋過最高塔的安全空域 (m)
        self.retreat_height  = 0.22   # 退避高度（放大至 0.22m，徹底解決撤離撞到既有方塊的問題）

        # ── 狀態機內部計步器 ──────────────────────────────────────────────────
        self._event = 0               # 當前所在的 Phase 編號 (0~4)
        self._step  = 0               # 當前 Phase 的步數計數

        # ── 軌跡插值輔助變數 ──────────────────────────────────────────────────
        self._phase_start_position = None   # 每個 Phase 起始時鎖定的末端位置
        self._lift_target_position = None   # Phase 1 V 形頂部目標位置（安全高度）
        self._retreat_target_position = None  # Phase 4 退避目標位置
        self._grasp_close_start_step: int | None = None

        # ── 任務完成旗標 ──────────────────────────────────────────────────────
        self._pose_target_failed = False    # 偵測到放置位姿未達標的錯誤旗標

    def _scale_steps(self, steps: int | float) -> int:
        """Convert a nominal simulation-frame duration to the selected playback speed."""
        return max(1, int(round(float(steps) / self.playback_speed)))

    # ──────────────────────────────────────────────────────────────────────────
    # 軌跡插值核心：S 曲線 (Smoothstep)
    # ──────────────────────────────────────────────────────────────────────────

    def _smoothstep(self, step: int, total_steps: int) -> float:
        """三次 Hermite 平滑插值比例 s(t) = 3t²－2t³，t ∈ [0, 1]。

        推導：令三次式 s(t)=at³+bt²+ct+d，以 s(0)=0、s'(0)=0、
        s(1)=1、s'(1)=0 作為端點位置與斜率條件，可得 d=c=0、
        a+b=1、3a+2b=0，解出 a=-2、b=3。因此 s'(0)=s'(1)=0。
        對直線位置命令 p(t)=p0+s(t)(p1-p0)，有
        p'(t)=s'(t)(p1-p0)，所以命令軌跡兩端的參數速度為零；這不等於保證
        物理手臂的實際速度、加速度或接觸力也連續。
        """
        t = min(max(step / max(total_steps, 1), 0.0), 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _set_phase_pose(
        self,
        target_position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str,
        *,
        locked_dof_indices: list[int] | None = None,
        position_only: bool = False,
        step_override: int | None = None,
        total_override: int | None = None,
    ) -> None:
        """以 S 曲線產生本影格的末端位置命令，再交由 Franka dIK 轉成關節目標。

        p_start 是進入該 Phase 時讀取的末端實際世界位置；p_target 是任務層提供
        的世界座標位置。每影格命令為 p_start + (p_target - p_start) * s(t)。
        orientation 使用 [w, x, y, z] 四元數。這裡只建立單影格笛卡兒目標，
        關節增量由 Franka.set_end_effector_pose() 使用當前 Jacobian 計算。

        參數：
            target_position: 本階段末端目標世界位置，形狀可轉為 (3,)。
            orientation: 末端目標四元數 [w, x, y, z]。
            ik_method: 傳給 Franka dIK 的求解器名稱。
            locked_dof_indices: 不參與本次 dIK 的手臂關節索引。
            position_only: 為 True 時只使用 Jacobian 的平移列。
            step_override: 子階段專用的插值步數；未提供則使用 FSM 步數。
            total_override: 子階段專用的總步數；未提供則使用 Phase 設定。
        """
        step  = self._step  if step_override  is None else step_override
        total = max(1, int(self.events_dt[self._event])) if total_override is None else total_override
        target_pos = np.asarray(target_position, dtype=np.float32).reshape(3)

        # 每個 Phase 第 0 步鎖定當前末端實際位置作為軌跡起點
        if self._phase_start_position is None:
            _, curr_pos, _ = self.robot.get_current_state()
            self._phase_start_position = curr_pos[0].copy()

        s = self._smoothstep(step + 1, total)
        cmd_position = self._phase_start_position + (target_pos - self._phase_start_position) * s

        self.robot.set_end_effector_pose(
            position=cmd_position,
            orientation=orientation,
            ik_method=ik_method,
            locked_dof_indices=locked_dof_indices,
            position_only=position_only,
            delta_scale=1.0,
        )

    def _advance_phase(self, target_position: np.ndarray, tolerance: float = 0.012) -> None:
        """以實際末端位置和模擬影格數判斷是否切換 Phase（精度細化至原本 2 倍）。

        Phase 0–2 在走過額定時間 70% 且位置誤差小於 tolerance (12mm) 時可提早結束；
        Phase 3、4 必須至少走完額定影格並分別達 5 mm、8 mm 高精度門檻。任何 Phase
        超過按播放速度縮放的逾時影格仍未收斂時會強制前進，避免控制器永久卡住。
        """
        self._step += 1
        phase_duration = max(1, int(self.events_dt[self._event]))
        _, current_pos, _ = self.robot.get_current_state()
        err = float(np.linalg.norm(current_pos - np.asarray(target_position).reshape(1, 3)))

        # Phase 3 和 Phase 4 必須完整執行至收斂，不允許提早跳出
        if self._event not in (3, 4):
            early_step = max(self._scale_steps(6), int(phase_duration * 0.70))
            if self._step >= early_step and err <= tolerance:
                self._next_phase()
                return

        if self._step >= phase_duration:
            # 各 Phase 精度細化至原本 2 倍高精度收斂門檻：
            if   self._event == 3: thresh = 0.005  # 垂直下放：5mm 嚴格觸地（原 10mm 細化 2 倍）
            elif self._event == 4: thresh = 0.008  # 退避：8mm 高精保靜止（原 15mm 細化 2 倍）
            else:                  thresh = tolerance
            if err <= thresh:
                self._next_phase()
            elif self._step >= phase_duration + self._scale_steps(35):
                # 超時強制推進（逾時保護，防止動畫卡死）
                self._next_phase()

    def _next_phase(self) -> None:
        """切換至下一個 Phase，重置步數與起點記憶。"""
        self._event += 1
        self._step  = 0
        self._phase_start_position = None

    # ──────────────────────────────────────────────────────────────────────────
    # 場景建立
    # ──────────────────────────────────────────────────────────────────────────

    def setup_scene(
        self,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        cube_size: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        robot_path: str = "/World/robot",
        cube_path: str = "/World/Cube",
    ) -> None:
        """在目前 USD Stage 建立 Franka、基礎方塊及必要環境。

        本專案透過 NVIDIA Isaac Sim Cube、GeomPrim、GroundPlane、DomeLight
        等 API 建立與包裝場景物件。所有位置、偏移和尺寸均以世界座標／公尺表達；
        子類別可傳入自己的來源方塊位置、尺寸與 Prim 路徑。

        Args:
            cube_initial_position: 方塊初始世界位置；未提供時使用預設值。
            cube_initial_orientation: 初始四元數 [w, x, y, z]。
            cube_size: 方塊 XYZ 尺寸（公尺）。
            target_position: 預設末端目標世界位置（公尺）。
            offset: 同時套用到方塊、目標及機器人根節點的世界平移。
            robot_path: Franka USD 在 Stage 中的 Prim 路徑。
            cube_path: 基礎方塊在 Stage 中的 Prim 路徑。
        """
        self.cube_initial_position = (
            cube_initial_position
            if cube_initial_position is not None
            else np.array([0.5, 0.0, 0.0258])
        )
        self.cube_initial_orientation = (
            cube_initial_orientation
            if cube_initial_orientation is not None
            else np.array([1, 0, 0, 0])
        )
        self.target_position = (
            target_position
            if target_position is not None
            else np.array([0.0, 0.5, 0.14])
        )
        self.cube_size = (
            cube_size if cube_size is not None else np.array([0.0515, 0.0515, 0.0515])
        )
        self.offset = offset if offset is not None else np.array([0.0, 0.0, 0.0])

        self.cube_initial_position += self.offset
        self.target_position       += self.offset

        # ── 建立 Franka 手臂 ──────────────────────────────────────────────────
        self.robot = Franka(robot_path=robot_path, create_robot=True)
        self.end_effector_link = self.robot.end_effector_link
        self.robot.set_world_poses(
            positions=[self.offset[0], self.offset[1], 0.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
        )

        # ── 建立地面與環境光 ──────────────────────────────────────────────────
        stage = stage_utils.get_current_stage()
        if not stage.GetPrimAtPath("/World/ground_plane"):
            GroundPlane("/World/ground_plane")
            DomeLight("/World/DomeLight").set_intensities(1000)

        # ── 建立測試用方塊剛體 ────────────────────────────────────────────────
        cube_shape = Cube(
            paths=cube_path,
            positions=self.cube_initial_position,
            orientations=self.cube_initial_orientation,
            sizes=1.0,
            scales=self.cube_size,
            colors="blue",
        )
        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
        self.cube = RigidPrim(paths=cube_shape.paths)

    # ──────────────────────────────────────────────────────────────────────────
    # 主循環：5 階段 FSM 步進
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, ik_method: str = "damped-least-squares") -> bool:
        """依目前 FSM 階段執行一影格取放控制。

        呼叫端負責推進 Isaac Sim 模擬影格。本方法只下達關節目標並更新 FSM 計數，
        不自行呼叫 SimulationApp.update()。呼叫方可藉 is_done() 判斷五個 Phase
        是否已完成；Hanoi 子類別會在此基礎上插入盤子抬升與放置校驗。

        Args:
            ik_method: Franka 微分逆運動學求解方法。

        Returns:
            FSM 尚在執行時回傳 True；本方法進入時若已完成則回傳 False。
        """
        if self.is_done():
            return False

        goal_orientation = self.robot.get_downward_orientation()

        # ──────────────────────────────────────────────────────────────────────
        # Phase 0：橫向巡航至方塊正上方（夾爪全開）
        # ──────────────────────────────────────────────────────────────────────
        if self._event == 0:
            self.robot.open_gripper()

            cube_pos = self.cube.get_world_poses()[0].numpy()
            safe_z = max(
                float(cube_pos[0, 2]) + self.pregrasp_height,
                getattr(self, '_safe_transit_z', lambda: 0.28)(),
            )
            goal_pos = np.array([cube_pos[0, 0], cube_pos[0, 1], safe_z])
            self._set_phase_pose(goal_pos, goal_orientation, ik_method)
            self._advance_phase(goal_pos)

        # ──────────────────────────────────────────────────────────────────────
        # Phase 1：V 形抓取＋退避到安全高度
        #
        # 下潛後張爪等待末端到位，再閉爪並提升：
        #   sub-phase A（前 1/3）：S 曲線從安全高度下潛至 approach_height
        #   sub-phase B：在底部張爪等待末端收斂，再閉爪保持
        #   sub-phase C：S 曲線從 approach_height 提升回安全高度
        #
        # 底部等待末端位置收斂，再閉爪保持後提升；這段等待用來減少過早夾爪造成的抓偏。
        # ──────────────────────────────────────────────────────────────────────
        elif self._event == 1:
            total = max(3, int(self.events_dt[1]))
            descent_steps = total // 3
            settle_min_steps = max(self._scale_steps(8), total // 8)
            settle_max_steps = max(settle_min_steps, total // 3)
            close_steps = max(self._scale_steps(12), total // 3)
            lift_steps = max(1, total // 3)

            # ── 在 Phase 1 第 0 步：一次性鎖定安全起點與計算各段目標 ──────
            if self._step == 0:
                _, curr_pos, _ = self.robot.get_current_state()
                self._phase_start_position = curr_pos[0].copy()   # 起點：當前安全高度
                self._grasp_close_start_step = None

                # 抓取目標由實際盤面頂部加安全間距決定，不隨盤數改變積木厚度。
                cube_pos = self.cube.get_world_poses()[0].numpy()[0]
                self._v_bottom_pos = np.array([
                    cube_pos[0],
                    cube_pos[1],
                    cube_pos[2] + self.cube_size[2] / 2 + self.approach_height,
                ])
                # 計算 V 形頂部（提升目標）：安全巡航高度
                safe_z = max(
                    self._phase_start_position[2],                             # 起點高度
                    getattr(self, '_safe_transit_z', lambda: 0.28)(),          # 動態安全走廊
                )
                self._lift_target_position = self._phase_start_position.copy()
                self._lift_target_position[2] = safe_z

            # ── sub-phase A：下潛並保持張爪 ────────────────────────────────
            if self._step < descent_steps:
                # 以當前 Phase 起點 → V 底部插值
                s = self._smoothstep(self._step + 1, descent_steps)
                cmd = self._phase_start_position + (self._v_bottom_pos - self._phase_start_position) * s
                self.robot.set_end_effector_pose(
                    position=cmd, orientation=goal_orientation, ik_method=ik_method,
                )
                self.robot.open_gripper()

            # ── sub-phase B：在底部張爪等待末端到位 ────────────────────────
            elif self._grasp_close_start_step is None:
                # 位置命令固定在 V 底部，夾爪維持張開；位置收斂或等待上限到達後才開始閉爪。
                self.robot.set_end_effector_pose(
                    position=self._v_bottom_pos, orientation=goal_orientation, ik_method=ik_method,
                )
                self.robot.open_gripper()
                _, current_position, _ = self.robot.get_current_state()
                position_error = float(np.linalg.norm(current_position[0] - self._v_bottom_pos))
                settle_steps = self._step - descent_steps
                if (
                    settle_steps >= settle_min_steps and position_error <= 0.005
                ) or settle_steps >= settle_max_steps:
                    self._grasp_close_start_step = self._step + 1

            # ── sub-phase C：閉爪保持後再提升退避 ──────────────────────────
            elif self._step < self._grasp_close_start_step + close_steps:
                self.robot.set_end_effector_pose(
                    position=self._v_bottom_pos, orientation=goal_orientation, ik_method=ik_method,
                )
                self.robot.close_gripper()
            else:
                # 以 V 底部 → 安全巡航高度插值（起點固定為 V 底）
                lift_start_step = self._grasp_close_start_step + close_steps
                local_step = self._step - lift_start_step
                local_total = lift_steps
                s = self._smoothstep(local_step + 1, local_total)
                cmd = self._v_bottom_pos + (self._lift_target_position - self._v_bottom_pos) * s
                self.robot.set_end_effector_pose(
                    position=cmd, orientation=goal_orientation, ik_method=ik_method,
                )
                self.robot.close_gripper()

            # ── Phase 1 推進（走完全部步數才切換）──────────────────────────
            self._step += 1
            if (
                self._grasp_close_start_step is not None
                and self._step >= self._grasp_close_start_step + close_steps + lift_steps
            ):
                self._next_phase()

        # ──────────────────────────────────────────────────────────────────────
        # Phase 2：水平高空走廊搬運至目標柱正上方（S 曲線）
        # ──────────────────────────────────────────────────────────────────────
        elif self._event == 2:
            safe_carry_z = max(
                float(self._lift_target_position[2]),
                getattr(self, '_safe_transit_z', lambda: 0.28)(),
            )
            goal_pos = self._lift_target_position.copy()
            goal_pos[:2] = np.asarray(self.target_position)[:2]
            goal_pos[2]  = safe_carry_z

            self._set_phase_pose(goal_pos, goal_orientation, ik_method)
            self._advance_phase(goal_pos)

        # ──────────────────────────────────────────────────────────────────────
        # Phase 3：垂直緩降落至放置位置（S 曲線 + 嚴格 10mm 收斂）
        # ──────────────────────────────────────────────────────────────────────
        elif self._event == 3:
            self._set_phase_pose(self.target_position, goal_orientation, ik_method)
            self._advance_phase(self.target_position)

        # ──────────────────────────────────────────────────────────────────────
        # Phase 4：開爪釋放 + 垂直退避回安全高度（S 曲線）
        #
        # 內部兩段：
        #   前 1/4 步：保持當前位置並開爪（讓方塊在物理引擎中穩定落地）
        #   後 3/4 步：S 曲線從釋放位置平滑退避至安全高度
        # ──────────────────────────────────────────────────────────────────────
        elif self._event == 4:
            total    = max(self._scale_steps(4), int(self.events_dt[4]))
            open_seg = max(self._scale_steps(2), total // 4)   # 前 1/4 步用於開爪靜置

            if self._step == 0:
                _, curr_pos, _ = self.robot.get_current_state()
                safe_z = max(
                    float(curr_pos[0, 2]) + self.retreat_height,
                    getattr(self, '_safe_transit_z', lambda: 0.28)(),
                )
                self._retreat_target_position = curr_pos[0].copy()
                self._retreat_target_position[2] = safe_z
                # 鎖定退避起點（開爪靜置段的位置）
                self._retreat_start_position = curr_pos[0].copy()

            # 前段：開爪靜止在放置位置（讓物理接觸穩定）
            if self._step < open_seg:
                self.robot.open_gripper()
                self.robot.set_end_effector_pose(
                    position=self._retreat_start_position,
                    orientation=goal_orientation,
                    ik_method=ik_method,
                )
            # 後段：S 曲線退避（起點 = 釋放位置，終點 = 安全高度）
            else:
                self.robot.open_gripper()
                local_step  = self._step - open_seg
                local_total = max(1, total - open_seg)
                s = self._smoothstep(local_step + 1, local_total)
                cmd = self._retreat_start_position + \
                      (self._retreat_target_position - self._retreat_start_position) * s
                self.robot.set_end_effector_pose(
                    position=cmd, orientation=goal_orientation, ik_method=ik_method,
                )

            # Phase 4 也需走完才切換（確保退避完全靜止，消除放後擺動）
            self._step += 1
            if self._step >= total:
                # 以退避目標位置做最終誤差確認
                _, curr_pos, _ = self.robot.get_current_state()
                err = float(np.linalg.norm(
                    curr_pos - np.asarray(self._retreat_target_position).reshape(1, 3)
                ))
                if err <= 0.015 or self._step >= total + self._scale_steps(40):
                    self._next_phase()

        return True

    # ──────────────────────────────────────────────────────────────────────────
    # 狀態查詢與重置
    # ──────────────────────────────────────────────────────────────────────────

    def is_done(self) -> bool:
        """回傳 True 表示所有 5 個 Phase 均已完成。"""
        return self._event >= len(self.events_dt)

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ) -> None:
        """重置機器人與方塊位姿至初始狀態。"""
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self) -> None:
        """將手臂重置至預設關節姿態，並清空狀態機所有計步。"""
        if self.robot is not None:
            self.robot.reset_to_default_pose()
        self._event = 0
        self._step  = 0
        self._phase_start_position    = None
        self._lift_target_position    = None
        self._retreat_target_position = None

    def reset_cube(
        self,
        position: np.ndarray | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        """將方塊重置至指定位置（預設為初始位置）。"""
        if self.cube is not None:
            pos = position    if position    is not None else self.cube_initial_position
            ori = orientation if orientation is not None else self.cube_initial_orientation
            self.cube.set_world_poses(positions=pos, orientations=ori)
