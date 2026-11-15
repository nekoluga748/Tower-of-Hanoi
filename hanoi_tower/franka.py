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

"""Franka Emika Panda 7 軸機械手臂與雙指夾爪控制器。

本檔保留 NVIDIA Copyright 與 Apache-2.0 授權標頭。它透過 Isaac Sim 的
Articulation／RigidPrim 介面操作 NVIDIA Franka USD 資產，並在專案端計算
微分逆運動學（dIK）。上層提供末端世界座標目標；本模組讀取當前關節、末端姿態與
Jacobian，計算本影格的關節位置增量，再呼叫 NVIDIA Articulation API 設定關節目標。

Hanoi 規劃不在本檔；tower.py 選擇當前盤與目標柱，pick_place.py 產生逐影格末端位置。
Warp 用於四元數運算，並不代表整個 dIK 或 PhysX 模擬都由 Warp 執行。
"""

from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.experimental.utils.impl.transform import quaternion_conjugate, quaternion_multiplication
from isaacsim.storage.native import get_assets_root_path


class Franka(Articulation):
    """Franka Panda 機械手臂控制器。

    繼承 NVIDIA Isaac Sim Articulation，負責手臂關節目標、末端 panda_hand 位姿
    讀取及平行夾爪開闔。預設載入 NVIDIA 資產庫中的 Franka Panda USD；dIK、零空間
    姿態偏好及控制器的目標轉換由本專案程式實作。
    """

    def __init__(
        self,
        robot_path: str = "/World/robot",
        create_robot: bool = True,
        end_effector_link: RigidPrim | None = None,
    ) -> None:
        """初始化手臂模型與末端夾爪。

        參數：
            robot_path: Franka 根 Prim 在 USD Stage 的路徑。
            create_robot: 為 True 時從 NVIDIA Isaac Sim 資產根目錄引用官方 Franka USD。
            end_effector_link: 自訂末端剛體；未提供時使用 {robot_path}/panda_hand。

        USD 變體選擇 AlternateFinger 夾爪與 Performance 網格。機器人模型及
        Isaac Sim Articulation API 是 NVIDIA 外部資源；下方 dIK 與目標轉換則是專案程式。
        """
        if create_robot:
            # 從 Isaac Sim 雲端/本機資產庫載入 Franka 模型，指定 AlternateFinger 夾爪與高效能 Mesh
            stage_utils.add_reference_to_stage(
                usd_path=get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                path=robot_path,
                variants=[("Gripper", "AlternateFinger"), ("Mesh", "Performance")],
            )

        # 初始化底層 Articulation 關節結構
        super().__init__(robot_path)

        # 設定末端執行器剛體 Prim
        if end_effector_link is None:
            self.end_effector_link = RigidPrim(f"{robot_path}/panda_hand")
        else:
            self.end_effector_link = end_effector_link

        if create_robot:
            # 設定初始預設關節姿態（前 7 個為手臂旋轉軸，後 2 個為夾爪平移指，0.04m 代表張開 80mm）
            self.set_default_state(dof_positions=[0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04])

        # 零空間偏好姿態（讓手肘自然朝下，避免關節奇異點與奇形怪狀的扭轉）
        self.default_arm_positions = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741])
        self.end_effector_link_index = self.get_link_indices("panda_hand").list()[0]

        # 夾爪行程極限（Franka 原廠平行指行程：每指 0.04m，總張開寬度 0.08m = 80mm）
        self.gripper_open_position = [0.04, 0.04]
        self.gripper_closed_position = [0.0, 0.0]

    def differential_inverse_kinematics(
        self,
        jacobian_end_effector: np.ndarray,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray | None = None,
        method: str = "damped-least-squares",
        method_cfg: dict[str, float] = None,
        position_only: bool = False,
        current_dof_positions: np.ndarray | None = None,
        default_dof_positions: np.ndarray | None = None,
        null_space_gain: float = 0.02,
    ) -> np.ndarray:
        """以目前 Jacobian 將末端笛卡兒誤差轉成一影格的關節位置增量。

        在目前構型 q 附近，正向運動學的一階展開為
        x(q + Δq) ≈ x(q) + J(q)Δq；令 e = x_goal - x(q)，就得到
        JΔq ≈ e。這只在小增量附近成立，因此回傳單次增量，並由呼叫端重新量測
        狀態後迭代，不是一次求出整段軌跡。

        Franka 全姿態控制時，J 為 6×7、Δq 為 7×1、e 為 6×1：

            [Δx]       [Δq1]
            [Δy]       [Δq2]
            [Δz]       [Δq3]
        J * [Δrx]  = e [Δq4]
            [Δry]       [Δq5]
            [Δrz]       [Δq6]
                         [Δq7]

        手算示例（僅用平面二連桿說明，不是 Franka 的實測值）：兩桿長皆為 1 m，
        q1=0、q2=90 度時，末端位置為 (1, 1) m，Jacobian 為

            J = [ -1  -1 ]       目標 (1.1, 0.9) m
                [  1   0 ]       e = (0.1, -0.1) m

        解 J*Δq=e：第二列給 Δq1=-0.1 rad；代入第一列得 Δq2=0 rad。
        這只是局部線性預測；將新角度代回原非線性正向運動學，位置約為
        (1.0948, 0.8952) m，仍有約 7 mm 誤差。下一影格重新量測並重算 Jacobian，
        就是在逐次修正這個線性化殘差。實際 Franka 的 6×7 Jacobian 由 Isaac Sim
        依目前關節構型提供，沒有一組固定數值可套用於所有姿態。

        本方法依 position_only 使用 3 維位置誤差，
        或把平移誤差與四元數方向誤差合成 6 維向量，再以指定求解器計算 dq。
        有冗餘關節時，額外加入 (I - J⁺J)(q_default - q_current) 的零空間偏好項，
        使姿態朝舒適構型移動。回傳的是單次迭代增量，不是整段路徑，也不保證一次到位；
        上層每個模擬影格重新讀取狀態並重算目標。

        Args:
            jacobian_end_effector: 末端 Jacobian，形狀為 batch × task 維度 × 活躍關節數。
            current_position: 目前末端世界位置，形狀為 batch × 3。
            current_orientation: 目前末端四元數 [w, x, y, z]。
            goal_position: 目標末端位置，形狀為 batch × 3。
            goal_orientation: 目標四元數；未提供時維持目前方向。
            method: damped-least-squares、pseudoinverse、transpose 或 SVD 求解器。
            method_cfg: 求解器縮放、阻尼與奇異值門檻設定。
            position_only: 是否只解三維平移誤差。
            current_dof_positions: 活躍關節目前角度。
            default_dof_positions: 零空間姿態偏好關節角。
            null_space_gain: 零空間偏好增益。

        Returns:
            與活躍關節順序相同的關節位置增量陣列。
        """
        if method_cfg is None:
            method_cfg = {"scale": 1.0, "damping": 0.05, "min_singular_value": 1e-5}

        scale = method_cfg.get("scale", 1.0)
        if position_only:
            # 僅控制三維位置 (X, Y, Z)
            error = np.expand_dims(goal_position - current_position, axis=2)
            if method == "transpose":
                pseudoinverse = np.swapaxes(jacobian_end_effector, 1, 2)
                delta_q = (scale * pseudoinverse @ error).squeeze(-1)
            elif method == "damped-least-squares":
                delta_q, pseudoinverse = self._adaptive_damped_least_squares(
                    jacobian_end_effector, error, scale, method_cfg
                )
            else:
                pseudoinverse = np.linalg.pinv(jacobian_end_effector)
                delta_q = (scale * pseudoinverse @ error).squeeze(-1)

            # 以精確 Moore-Penrose 偽逆建立零空間投影；在此局部模型下，
            # J(I - J+J) = 0，因此次要姿態修正不改變主要位置任務的一階輸出。
            if current_dof_positions is not None and default_dof_positions is not None:
                num_dofs = jacobian_end_effector.shape[-1]
                eye = np.eye(num_dofs, dtype=np.float32)
                pinv_exact = np.linalg.pinv(jacobian_end_effector)
                null_proj = eye - pinv_exact @ jacobian_end_effector
                dof_diff = np.expand_dims(default_dof_positions - current_dof_positions, axis=-1)
                null_delta = (null_proj @ dof_diff).squeeze(-1) * null_space_gain
                delta_q = delta_q + null_delta
            return delta_q

        # 同時控制位置與朝向姿態 (6 自由度)
        goal_orientation = current_orientation if goal_orientation is None else goal_orientation

        # Warp 與 Isaac Sim 四元數工具負責計算相對旋轉；其餘 Jacobian 求解在此以 NumPy 執行。
        goal_quat_wp = wp.from_numpy(goal_orientation, dtype=wp.float32)
        current_quat_wp = wp.from_numpy(current_orientation, dtype=wp.float32)

        # 計算四元數誤差：q_error = q_goal * q_current_conjugate
        # 單位四元數的共軛是逆旋轉，因此此乘積表示由目前方向轉到目標方向的相對旋轉。
        current_quat_conjugate_wp = quaternion_conjugate(current_quat_wp)
        q_wp = quaternion_multiplication(goal_quat_wp, current_quat_conjugate_wp)
        q_np = q_wp.numpy()

        # 處理四元數雙重覆蓋 (Double Cover, 取短路徑旋轉)
        # q 與 -q 表示同一旋轉；令相對四元數標量部非負，可選到角度不超過 π 的表示。
        quaternion_sign = np.where(q_np[:, [0]] < 0.0, -1.0, 1.0)
        orientation_error = q_np[:, 1:] * quaternion_sign

        # 將世界位置差與相對四元數向量部份合成 batch × 6 × 1 的任務空間誤差。
        # 若相對旋轉為 (cos(θ/2), sin(θ/2)u)，其向量部是 sin(θ/2)u，
        # 小角度時約等於 (θ/2)u，因而可作為局部三維姿態誤差；它不是四元數直接相減。
        error = np.expand_dims(
            np.concatenate([goal_position - current_position, orientation_error], axis=-1), axis=2
        )

        # 根據指定逆運動學求解器計算 delta_q
        if method == "singular-value-decomposition":
            # 奇異值分解 SVD
            min_singular_value = method_cfg.get("min_singular_value", 1e-5)
            U, S, Vh = np.linalg.svd(jacobian_end_effector)
            inv_s = np.where(min_singular_value < S, 1.0 / S, np.zeros_like(S))
            pseudoinverse = np.swapaxes(Vh, 1, 2)[:, :, :6] @ np.diagflat(inv_s) @ np.swapaxes(U, 1, 2)
            delta_q = (scale * pseudoinverse @ error).squeeze(-1)
        elif method == "pseudoinverse":
            # Moore-Penrose 廣義逆矩陣
            pseudoinverse = np.linalg.pinv(jacobian_end_effector)
            delta_q = (scale * pseudoinverse @ error).squeeze(-1)
        elif method == "transpose":
            # 雅可比轉置矩陣法
            pseudoinverse = np.swapaxes(jacobian_end_effector, 1, 2)
            delta_q = (scale * pseudoinverse @ error).squeeze(-1)
        elif method == "damped-least-squares":
            # 奇異值自適應阻尼最小平方法：抑制近奇異方向的小奇異值增益。
            delta_q, pseudoinverse = self._adaptive_damped_least_squares(
                jacobian_end_effector, error, scale, method_cfg
            )
        else:
            raise ValueError(f"無效的 IK 方法：{method}")

        # 零空間次要目標：N = I - J⁺J 將關節增量投影到 J 的零空間，因為
        # JN = J - JJ⁺J = 0（Moore-Penrose 偽逆滿足 JJ⁺J=J）。所以加入
        # N(q_default-q) 不改變主要任務的一階末端增量；有限步長下 J 會隨構型變化，
        # 故不保證非線性位姿完全不變。
        if current_dof_positions is not None and default_dof_positions is not None:
            num_dofs = jacobian_end_effector.shape[-1]
            eye = np.eye(num_dofs, dtype=np.float32)
            pinv_exact = np.linalg.pinv(jacobian_end_effector)
            null_proj = eye - pinv_exact @ jacobian_end_effector
            dof_diff = np.expand_dims(default_dof_positions - current_dof_positions, axis=-1)
            null_delta = (null_proj @ dof_diff).squeeze(-1) * null_space_gain
            delta_q = delta_q + null_delta

        return delta_q

    @staticmethod
    def _adaptive_damped_least_squares(
        jacobian: np.ndarray,
        error: np.ndarray,
        scale: float,
        method_cfg: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """以 SVD 計算批次化自適應阻尼最小平方法偽逆。

        DLS 是最小化 ||JΔq-e||² + λ²||Δq||² 的正則化最小平方解。
        對 J = UΣVᵀ，可得 Δq = V diag(σ/(σ²+λ²)) Uᵀe，亦即以
        σ/(σ²+λ²) 取代偽逆中的 1/σ；當 σ 趨近 0 時此增益也趨近 0，
        避免近奇異方向把微小誤差放大成過大的關節增量。
        最小奇異值低於 singularity_threshold 時，阻尼 λ 由 base 線性增加至 max；
        這可限制近奇異姿態的增量放大，但不等同完整的關節限位或碰撞安全規劃。

        Args:
            jacobian: batch × task 維度 × 關節數的 Jacobian。
            error: batch × task 維度 × 1 的末端誤差。
            scale: 解算增量縮放係數。
            method_cfg: damping、max_damping 和 singularity_threshold 設定。

        Returns:
            (delta_q, pseudoinverse)，分別為關節增量及當前阻尼偽逆。
        """
        base_damping = method_cfg.get("damping", 0.05)
        max_damping = method_cfg.get("max_damping", 0.25)
        singularity_threshold = method_cfg.get("singularity_threshold", 0.15)

        # J = U diag(sigma) V^T；奇異值描述各正交運動方向上的局部增益。
        left_vectors, singular_values, right_vectors = np.linalg.svd(jacobian, full_matrices=False)
        smallest_singular_value = singular_values[:, -1]

        # 最小奇異值高於門檻時使用 base_damping；低於門檻時線性提高阻尼，
        # 並在 max_damping 飽和，避免倒數增益在近奇異構型中無界放大。
        singularity_ratio = np.clip(
            (singularity_threshold - smallest_singular_value) / singularity_threshold,
            0.0,
            1.0,
        )
        damping = base_damping + (max_damping - base_damping) * singularity_ratio

        # 阻尼偽逆因子：σ / (σ² + λ²)，避免小奇異值被直接取倒數而過度放大。
        inverse_factors = singular_values / (singular_values**2 + damping[:, None] ** 2)
        pseudoinverse = np.einsum(
            "bik,bk,bkj->bij",
            np.swapaxes(right_vectors, 1, 2),
            inverse_factors,
            np.swapaxes(left_vectors, 1, 2),
        )
        return (scale * pseudoinverse @ error).squeeze(-1), pseudoinverse

    def get_current_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """讀取目前關節角與 panda_hand 的世界位姿。

        返回：
            (dof_positions, end_effector_position, end_effector_orientation)，分別為
            Articulation 關節位置、末端世界座標 XYZ，以及 [w, x, y, z] 世界四元數。
            這些資料來自 Isaac Sim 當前模擬狀態，不是控制命令或規劃目標。
        """
        current_dof_positions = self.get_dof_positions().numpy()
        current_end_effector_position, current_end_effector_orientation = self.end_effector_link.get_world_poses()
        current_end_effector_position = current_end_effector_position.numpy()
        current_end_effector_orientation = current_end_effector_orientation.numpy()

        return current_dof_positions, current_end_effector_position, current_end_effector_orientation

    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        ik_method: str = "damped-least-squares",
        locked_dof_indices: list[int] | None = None,
        position_only: bool = False,
        delta_scale: float = 1.0,
    ) -> None:
        """依末端目標位姿計算本影格的關節位置目標並下達給 Articulation。

        數學上先以目前狀態建立 e=x_goal-x_current，再由 Jacobian 求 Δq；最後
        q_command=q_current+Δq。全姿態模式的矩陣尺寸是 J(6×7)、e(6×1)、
        Δq(7×1)，位置模式則是 J(3×7)、e(3×1)。q_current 使用本影格實測關節角，
        而不是上一影格的命令，因此控制器會隨物理回饋重新計算。夾爪第 8、9 DOF
        不參與手臂 IK。求得的 q_command 是控制目標，不是已達成的實際姿態。

        本方法把目標送至 Isaac Sim 的 set_dof_position_targets()；Articulation／
        PhysX 再根據關節控制器與物理接觸推進實際狀態。

        參數：
            position: 末端目標世界位置 [x, y, z]，單位為公尺。
            orientation: 末端目標世界四元數 [w, x, y, z]。
            ik_method: dIK 求解方法。
            locked_dof_indices: 不參與此次 IK 的手臂 DOF 索引。
            position_only: 為 True 時忽略方向誤差，只使用位置 Jacobian。
            delta_scale: 關節增量縮放係數。
        """
        # 讀取當前關節與位姿
        current_dof_positions, current_end_effector_position, current_end_effector_orientation = (
            self.get_current_state()
        )

        position = np.asarray(position)
        orientation = np.asarray(orientation)
        if position.ndim == 1:
            position = position.reshape(1, -1)
        if orientation.ndim == 1:
            orientation = orientation.reshape(1, -1)

        # 由 NVIDIA Articulation 取得 Jacobian；link 索引轉為矩陣的零起算索引，只取手臂 7 軸。
        jacobian_matrices = self.get_jacobian_matrices().numpy()
        active_dof_indices = [index for index in range(7) if index not in (locked_dof_indices or [])]
        jacobian_end_effector = jacobian_matrices[:, self.end_effector_link_index - 1, :, :7]
        if position_only:
            jacobian_end_effector = jacobian_end_effector[:, :3, :]
        jacobian_end_effector = jacobian_end_effector[:, :, active_dof_indices]

        default_active = (
            self.default_arm_positions[active_dof_indices]
            if hasattr(self, "default_arm_positions")
            else None
        )
        if default_active is not None and default_active.ndim == 1:
            default_active = np.expand_dims(default_active, axis=0)

        # 求解 7 軸關節增量 delta_q
        delta_dof_positions = self.differential_inverse_kinematics(
            jacobian_end_effector=jacobian_end_effector,
            current_position=current_end_effector_position,
            current_orientation=current_end_effector_orientation,
            goal_position=position,
            goal_orientation=orientation,
            method=ik_method,
            position_only=position_only,
            current_dof_positions=current_dof_positions[:, active_dof_indices],
            default_dof_positions=default_active,
        )
        delta_dof_positions *= delta_scale

        # dIK 給的是本影格增量，故加回實測當前關節位置後才成為 position target。
        dof_position_targets = current_dof_positions[:, active_dof_indices] + delta_dof_positions
        self.set_dof_position_targets(dof_position_targets, dof_indices=active_dof_indices)

    def open_gripper(self) -> None:
        """將兩個夾爪 DOF 設為各 0.04 m，標稱總張開量 0.08 m。"""
        self.set_dof_position_targets(self.gripper_open_position, dof_indices=[7, 8])

    def close_gripper(self) -> None:
        """將兩個夾爪 DOF 目標設為 0；接觸力與是否抓牢由物理模擬決定。"""
        self.set_dof_position_targets(self.gripper_closed_position, dof_indices=[7, 8])

    def set_gripper_position(self, position: np.ndarray) -> None:
        """精確設定夾爪兩指間距。"""
        if position.ndim == 1:
            position = position.reshape(1, -1)
        self.set_dof_position_targets(position, dof_indices=[7, 8])

    def get_downward_orientation(self) -> np.ndarray:
        """回傳夾爪朝下的四元數 [w, x, y, z] = [0, 1, 0, 0]。"""
        return [0.0, 1.0, 0.0, 0.0]

    def reset_to_default_pose(self) -> None:
        """將機械手臂重置回初始開口安全待命姿態。"""
        default_positions = [0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, 0.04, 0.04]
        self.set_dof_positions(default_positions)
        self.set_dof_position_targets(default_positions)
