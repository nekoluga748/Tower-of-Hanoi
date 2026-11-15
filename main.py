"""NVIDIA Isaac Sim 中的 Franka 河內塔應用程式組合根。

本檔負責把模擬器生命週期、任務控制器及外部輸出組合起來；Hanoi 規則、抓放策略與
關節控制分屬不同層，避免任務邏輯直接依賴 USD／PhysX 細節：

* planner.py 是純 Python 規劃層，只產生 (盤子索引, 來源柱, 目標柱)，不讀寫場景。
* tower.py 是任務協調層，將符號移動映射成物件與末端目標，並在物理操作確認後提交
    _stacks；符號狀態代表已完成的任務，不取代剛體的實際位置。
* pick_place.py 是操作策略層，管理通用單物件抓放控制，不判斷 Hanoi 移動是否合法。
* franka.py 是機器人控制層，將末端位姿誤差和 Isaac Sim 提供的 Jacobian 轉成關節
    目標；PhysX／Articulation 負責實際運動，控制命令不等於物理結果。
* 本檔作為組合根，建立 Isaac Sim 與場景、驅動每幀更新，並處理錄影、結果輸出及關閉。

跨層的核心契約是「目標、命令、實測狀態各自分離」：規劃層提供離散意圖，控制層將意圖
轉為連續命令，任務協調層依實測剛體狀態驗證完成與否，再更新符號堆疊。這使演算法可
獨立測試，也避免尚未完成的物理操作被記成已完成的 Hanoi 步驟。

NVIDIA 外部資源與 API 在此入口的使用點：isaacsim.SimulationApp、omni.kit.app
擴充管理器、Timeline、SimulationManager、set_camera_view，以及錄影時使用的
Synthetic Data 感測器。河內塔規劃與控制邏輯由本專案實作；pandas 負責結果表格，
OpenCV 負責 MP4 編碼，兩者都不是 NVIDIA 資源。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

# =============================================================================
# 命令列參數解析 (CLI Arguments)
# =============================================================================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--headless", action="store_true", help="無圖形介面背景執行模式（適合伺服器/自動化測試）")
parser.add_argument("--test", action="store_true", help="相容舊用法；任務結束後皆會自動退出")
parser.add_argument("--disks", type=int, choices=range(2, 11), default=3, help="河內塔層數（支援 2 至 10 層，最大 1023 步）")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="物理模擬計算後端 (cpu 或 cuda)")
parser.add_argument(
    "--resolution",
    choices=("640x360", "960x540", "1280x720"),
    default="960x540",
    help="視窗與錄影解析度（預設 960x540）",
)
parser.add_argument(
    "--show-panels",
    action="store_true",
    help="保留 Isaac Sim 側邊與底部面板顯示（預設隱藏）",
)
parser.add_argument("--record", type=Path, help="將視口畫面儲存為 MP4 影片的路徑")
parser.add_argument("--record-fps", type=int, default=30, help="錄製影片的每秒幀率 (FPS)")
parser.add_argument(
    "--log-end-effector-pose",
    choices=("true", "false"),
    default="false",
    help="是否以 20 Hz 輸出末端位置與四元數姿態 (true/false)",
)
parser.add_argument(
    "--ik-method",
    choices=["singular-value-decomposition", "pseudoinverse", "transpose", "damped-least-squares"],
    default="damped-least-squares",
    help="逆運動學求解演算法（預設推薦 damped-least-squares，數值最穩定）",
)
args, _ = parser.parse_known_args()

if args.record_fps < 1:
    parser.error("--record-fps 必須至少為 1")

# 預設行為：若使用者未特別指定 --record，自動於 recordings/ 目錄建立帶有時間戳記的 MP4 錄影
if args.record is None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.record = Path(f"recordings/hanoi_{args.disks}disks_{timestamp}.mp4")

if args.record.suffix.lower() != ".mp4":
    parser.error("--record 輸出檔名後綴必須為 .mp4")

render_width, render_height = (int(value) for value in args.resolution.split("x"))

# SimulationApp 必須先於其他 Isaac Sim/Omniverse 匯入建立，讓 Kit 初始化 Python 執行環境。
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": render_width,
        "height": render_height,
        "window_width": render_width,
        "window_height": render_height,
    }
)
video_writer = None

try:
    import omni.kit.app
    import omni.ui as ui

    def hide_workspace_panels() -> None:
        for panel_name in (
            "Stage",
            "Property",
            "Content",
            "Console",
            "Timeline",
            "Render Settings",
        ):
            panel = ui.Workspace.get_window(panel_name)
            if panel is not None:
                panel.visible = False

    # NVIDIA Isaac Sim 擴充套件提供本專案使用的實驗性物件、Prim、材質及模擬管理 API。
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    for extension_name in (
        "isaacsim.core.experimental.objects",
        "isaacsim.core.experimental.materials",
        "isaacsim.core.experimental.prims",
        "isaacsim.core.experimental.utils",
        "isaacsim.core.simulation_manager",
        "isaacsim.storage.native",
    ):
        extension_manager.set_extension_enabled_immediate(extension_name, True)

    # NVIDIA Synthetic Data 感測器只在需要錄影時啟用，擷取視口 RGB 幀供 OpenCV 編碼。
    if args.record:
        extension_manager.set_extension_enabled_immediate("omni.syntheticdata", True)

    simulation_app.update()

    if not args.show_panels:
        # 在建立場景前收起停駐面板，讓模擬視口一開始就完整顯示。
        hide_workspace_panels()

    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    if args.record:
        import cv2
        import omni.syntheticdata._syntheticdata as synthetic_data
        from omni.kit.viewport.utility import get_active_viewport
        from omni.syntheticdata import sensors

    from hanoi_tower.tower import FrankaTowerOfHanoi

    # 配置物理模擬引擎運算裝置 (CPU / CUDA)
    SimulationManager.set_physics_sim_device(args.device)
    simulation_app.update()

    # 實例化河內塔控制器並在 USD Stage 中建立物件幾何與物理碰撞體
    controller = FrankaTowerOfHanoi(disk_count=args.disks)
    controller.setup_scene()
    simulation_app.update()
    if not args.show_panels:
        # 部分渲染面板可能在場景初始化期間建立，再隱藏一次確保不覆蓋視口。
        hide_workspace_panels()

    # 設定最佳觀察相機視角（居中斜俯視角，清楚呈現 3 根柱子與手臂姿態）
    set_camera_view(
        # 相機在世界座標中的位置 [x, y, z]
        eye=[1.13, -1.4, 1.0],
        # 相機朝向的目標點 [x, y, z]
        target=[0.15, 0.0, 0.20],
        camera_prim_path="/OmniverseKit_Persp",
    )

    # 綁定錄影视窗與 RGB 感測器管線
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("找不到可用的 Isaac Sim 視口以進行錄影")
        sensors.enable_sensors(viewport, [synthetic_data.SensorType.Rgb])
        simulation_app.update()

    # Timeline 控制模擬播放；SimulationManager.is_simulating() 決定何時推進控制器。
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()
    controller.reset()
    peg_names = ("A", "B", "C")
    completed_moves = []
    logged_move_count = 0
    last_pose_log_time = 0.0

    # =========================================================================
    # 模擬主循環 (Simulation Main Loop)
    # =========================================================================
    completed = False
    completion_time: float | None = None
    while simulation_app.is_running():
        # 每個模擬影格只送出一次控制更新；物理結果在後續 update 中回饋到剛體狀態。
        if SimulationManager.is_simulating() and not completed:
            controller.forward(args.ik_method)

        # _move_index 只在完整取放 FSM 完成後增加，每動完 1 step 即時輸出至終端機
        while logged_move_count < controller._move_index:
            disk_index, source, destination = controller._moves[logged_move_count]
            if logged_move_count == 0:
                print(" Step Disk Move", flush=True)
            step_num = logged_move_count + 1
            disk_name = f"D{disk_index + 1}"
            move_str = f"{peg_names[source]}->{peg_names[destination]}"
            print(f"{step_num:3d}    {disk_name:<2}  {move_str}", flush=True)
            completed_moves.append(
                {
                    "Step": step_num,
                    "Disk": disk_name,
                    "Move": move_str,
                }
            )
            logged_move_count += 1

        if controller.is_done() and not completed:
            completed = True
            completion_time = time.monotonic()
            moves_table = pd.DataFrame(completed_moves, columns=["Step", "Disk", "Move"])
            table_path = args.record.with_suffix(".csv")
            moves_table.to_csv(table_path, index=False, encoding="utf-8-sig")
            print(f"移動表已保存：{table_path}", flush=True)
            if controller._failed:
                print(f"任務中止，已完成 {len(completed_moves)} 步。", flush=True)
            else:
                print(f"河內塔完成，共移動 {len(completed_moves)} 步。", flush=True)

        # 更新物理世界與渲染畫面一幀
        simulation_app.update()

        now = time.monotonic()
        if args.log_end_effector_pose == "true" and now - last_pose_log_time >= 0.05:
            _, end_effector_position, end_effector_orientation = controller.robot.get_current_state()
            x, y, z = end_effector_position[0]
            qw, qx, qy, qz = end_effector_orientation[0]
            print(
                f"末端姿態位置 x={x:.3f} y={y:.3f} z={z:.3f} m, "
                f"四元數 w={qw:.3f} x={qx:.3f} y={qy:.3f} z={qz:.3f}",
                flush=True,
            )
            last_pose_log_time = now

        # NVIDIA 感測器提供 RGB；OpenCV 轉成 BGR 後寫入 MP4，這裡不疊加文字或檔名。
        if args.record and simulation_app.is_running():
            rgba = sensors.get_rgb(viewport)
            if rgba.ndim == 3 and rgba.shape[2] >= 3:
                height, width = rgba.shape[:2]
                if video_writer is None:
                    codec = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(args.record), codec, args.record_fps, (width, height)
                    )
                    if not video_writer.isOpened():
                        raise RuntimeError(f"無法建立 MP4 輸出檔：{args.record}")
                frame_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

        # 留一秒讓使用者看到結束狀態，並讓最後的錄影影格寫入後再關閉應用程式。
        if completed and completion_time is not None and time.monotonic() - completion_time >= 1.0:
            break
finally:
    # 釋放影片寫入器資源並關閉 SimulationApp
    try:
        if video_writer is not None:
            video_writer.release()
    finally:
        simulation_app.close()
