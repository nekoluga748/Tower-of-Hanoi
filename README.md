# 機械手臂河內塔虛實整合專案
## Franka Tower of Hanoi — Robotic Manipulation in NVIDIA Isaac Sim

---

## 🎯 核心學習目標

1. **場景建立**：由 `SimulationApp` 啟動 NVIDIA Isaac Sim，並在 USD Stage 動態建立多層河內塔場景（柱位、彩色剛體盤子及物理材質）；模擬播放由 Timeline 管理。
2. **河內塔演算法**：以遞迴分治法求得 $2^N - 1$ 步最優解，並透過符號層（內部堆疊狀態）與物理層（剛體位置）雙向同步，確保搬運順序合法無誤。
3. **手臂逆運動學**：實作差分逆運動學（dIK），以雅可比矩陣、阻尼最小平方法（Damped Least Squares）與零空間投影（Null-space Projection）驅動 Franka Emika Panda 7 軸手臂完成精確取放。

---

## 🏗️ 軟體架構與虛實整合

本專案直接以 `SimulationApp` 啟動 Isaac Sim，並透過 Timeline 與 `SimulationManager`
控制播放；沒有另外建立 `SimulationContext`、`World` 或 `InteractiveScene` 包裝層。
每一影格的控制與回饋順序如下：

| 階段 | 程式位置 | 輸入與輸出 |
|------|----------|------------|
| 啟動應用 | `main.py` | 建立 `SimulationApp`，啟用所需 Isaac Sim 擴充套件，啟動 Timeline |
| 建立虛擬場景 | `tower.py`、`pick_place.py` | 在 USD Stage 建立官方 Franka USD、地面、盤子剛體、柱位標記與材質 |
| 產生任務 | `planner.py`、`tower.py` | Hanoi 遞迴產生 `(盤子, 來源柱, 目標柱)`；`_stacks` 保存符號層柱位狀態 |
| 計算末端命令 | `pick_place.py`、`franka.py` | 以世界座標位置及朝下四元數建立目標，dIK 將笛卡兒誤差轉成 7 軸關節位置目標 |
| 物理回饋與提交 | `tower.py` | 從 `RigidPrim` 和 `panda_hand` 讀取實際世界位姿；抓取、搬運、放置完成後才更新符號堆疊 |
| 輸出與關閉 | `main.py` | 將已完成移動輸出至終端與 CSV；需要錄影時擷取 RGB 並交由 OpenCV 編碼 |

**虛實同步邊界：** `_stacks` 是任務規劃用的離散狀態，不是物理引擎中的物件位置。
盤子的實際位置由 Isaac Sim 剛體模擬；控制器在抓取確認和放置校正時讀取剛體世界位姿，
只有完整取放 FSM 完成才提交下一筆 Hanoi 移動。若抓取或放置檢查失敗，任務會重試或停止，
不會把尚未完成的移動記入結果表。

## 🧩 從 NVIDIA Franka 抓放示範到河內塔

NVIDIA Franka 抓取／放置示範提供「以末端目標控制 Franka，將一個物件由來源搬到目標」的
機器人操作模式。本專案把這個**單次 Pick-and-Place 任務單位**擴充成可重複執行的 Hanoi 遊戲：

1. **保留機器人與模擬介面：** `franka.py` 以 Isaac Sim Articulation 控制 7 個手臂關節及 2 個夾爪關節；`pick_place.py` 使用有限狀態機完成一次抓取、搬運與釋放。這兩個檔案保留 NVIDIA Copyright 與 Apache-2.0 SPDX 標頭。
2. **將固定物件改為多盤場景：** `tower.py` 依盤數建立不同尺寸的剛體盤子和三個柱位，並以 `self.cube` 指向目前規劃要搬的盤子；同一套 Pick-and-Place 控制器因此能重複操作不同剛體。
3. **加入符號任務規劃：** `planner.py` 使用經典遞迴產生最少移動序列；`tower.py` 以 `_stacks` 記錄每根柱子的盤子順序，計算本次來源與目的柱。
4. **把一次抓放接回 Hanoi 狀態機：** 每次抓放會檢查盤子是否確實抬起、目的位置是否對準；成功完成 5 個 Phase 後才更新 `_stacks` 和移動索引，再載入下一筆遞迴步驟。

因此，NVIDIA 範例提供的是機械臂模型、Isaac Sim 操作 API 及 Pick-and-Place 的控制基礎；
Hanoi 遞迴演算法、盤子尺寸規則、三柱符號狀態、重試／校正邏輯和任務紀錄則由本專案實作。
這不是把 NVIDIA 範例中的單一來源／目標位置直接改名，而是在其抓放單位之外加入任務規劃層與物理狀態協調層。

### NVIDIA 外部資源與 API

| 資源 | 本專案使用位置 | 用途／來源界線 |
|------|----------------|----------------|
| Franka Panda USD：`/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` | `franka.py` 的 `get_assets_root_path()` 與 `add_reference_to_stage()` | NVIDIA Isaac Sim 資產；選用 `AlternateFinger` 夾爪及 `Performance` 網格變體 |
| Isaac Sim `SimulationApp`、Timeline、`SimulationManager`、Extension Manager | `main.py` | NVIDIA Kit／Isaac Sim 應用生命週期、擴充管理與模擬狀態 API |
| `Cube`、`RigidPrim`、`GeomPrim`、`RigidBodyMaterial`、`GroundPlane`、`DomeLight` | `tower.py`、`pick_place.py` | NVIDIA Isaac Sim 實驗性物件、Prim、材質及場景 API；本專案用它們建立 USD 場景和物理剛體 |
| `set_camera_view`、Synthetic Data RGB Sensor | `main.py` | NVIDIA Isaac Sim 相機與視口影像擷取 API；只在錄影設定中使用 RGB 感測器 |
| NVIDIA Warp (`warp`) 與 Isaac Sim 四元數工具 | `franka.py` | Warp 陣列用於旋轉誤差計算；關節 Jacobian、dIK 分支及 Hanoi 規劃由本專案程式處理 |
| NVIDIA 官方文件 | [Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/) | API／資產使用參考；實際載入的機器人資產路徑以 `franka.py` 中的 USD 路徑為準 |

`planner.py` 不匯入 Isaac Sim，也不依賴 NVIDIA 演算法；河內塔的遞迴移動序列是本專案的純 Python 實作。

---

## 🔢 演算法說明

### a. 河內塔遞迴分治（符號推理層）

河內塔問題採用經典遞迴分治策略：

> 將 N-1 個盤子從來源柱移至輔助柱 → 將最大盤移至目標柱 → 再將 N-1 個盤子從輔助柱移至目標柱。

此策略可保證**最優步數為 $2^N - 1$**（N 為盤子數量）。本專案 `planner.py` 以純 Python 生成完整動作序列，支援 2～10 層（3 步至 1,023 步），並以內部堆疊（`_stacks`）維持符號層狀態，與 Isaac Sim 物理模擬層雙向同步。

---

### b. S 曲線平滑軌跡（三次 Hermite Smoothstep）

手臂各階段位置命令採用三次 Hermite 平滑插值函數：

$$s(t) = 3t^2 - 2t^3, \quad t \in [0, 1]$$

**物理意義：**
- $s(0) = 0$，$s(1) = 1$：端點準確抵達目標
- $s'(0) = 0$，$s'(1) = 0$：兩端速度為零，啟動與停止無衝擊
- 三次多項式的端點二階導數不為零，因此只主張端點速度為零，不宣稱端點加速度連續

相較於線性插值，smoothstep 讓笛卡兒位置命令以零端點速度進出；實際關節力矩和接觸結果仍由 Isaac Sim 物理控制器決定。抓取底部另有到位收斂等待，並非全程無停頓。

---

### c. 差分逆運動學（dIK）

每一模擬影格，控制器讀取末端目前世界位姿 $\mathbf{x}$、目前關節角 $\mathbf{q}$ 及
Isaac Sim Articulation 提供的末端 Jacobian $\mathbf{J}$。FSM 產生該影格的目標位姿
$\mathbf{x}_d$；程式計算局部誤差並解出關節增量 $\Delta\mathbf{q}$，再以
$\mathbf{q}_{cmd}=\mathbf{q}+\Delta\mathbf{q}$ 下達關節位置目標。下一影格重新讀取物理狀態，
所以這是閉迴路的逐影格差分 IK，不是一次產生整段關節軌跡。

概念式可寫為：

$$\Delta\mathbf{q} = \mathbf{J}^{+}\mathbf{e} + (\mathbf{I}-\mathbf{J}^{+}\mathbf{J})\mathbf{z}$$

其中 $\mathbf{e}$ 是位置誤差，或合併位置與方向誤差的 6 維向量；$\mathbf{z}$ 是回到
預設手肘姿態的零空間偏好。程式預設使用自適應 DLS，也可由 `--ik-method` 選擇其他方法。

| 項目 | 說明 |
|------|------|
| **雅可比矩陣** $\mathbf{J}$ | 關節空間到任務空間的線性映射，描述各關節對末端速度的貢獻 |
| **阻尼最小平方（DLS）** | 以奇異值 $\sigma$ 的 $\sigma/(\sigma^2+\lambda^2)$ 作阻尼逆；最小奇異值降低時增大 $\lambda$，限制增量放大，但不取代關節限位或碰撞檢查 |
| **零空間投影** | $(\mathbf{I} - \mathbf{J}^{+}\mathbf{J})\mathbf{z}$：使用 7 軸冗餘度偏向預設關節姿態；實作使用精確偽逆建立投影矩陣 |

`franka.py` 中的關節讀取、Jacobian、`set_dof_position_targets()`、`Articulation` 和
`RigidPrim` 來自 NVIDIA Isaac Sim；dIK 分支、阻尼調整及任務空間目標換算為本專案實作。

---

## 🤖 5 階段有限狀態機（FSM）

> **核心設計原則：1 個 Hanoi 步驟 = 5 個 Phase 的總和**
> 每搬移一片盤子，狀態機依序完整執行 Phase 0 → 1 → 2 → 3 → 4，完成後才進入下一步。

| Phase | 名稱 | 說明 |
|:-----:|------|------|
| **0** | 橫向巡航至方塊正上方 | 手臂保持安全巡航高度（`_safe_transit_z`），以 S 曲線水平移動至目標方塊正上方，避免碰撞柱上既有方塊 |
| **1** | V 形抓取＋退避 | **張爪下潛**至抓取目標 → 在底部張爪等待末端位置收斂（最長等待受限）→ **閉爪保持** → 提升至安全高度；抬升後再驗證盤子是否被夾起，失敗最多重試 3 次 |
| **2** | 水平搬運 | 夾持方塊以 S 曲線水平飛行至目標柱正上方；途中持續監測方塊相對夾爪滑移量 |
| **3** | 垂直緩降（10 mm 嚴格收斂） | 以精確 Z 軸下降至放置位置，注入 −6 mm 貼地微壓補償重力下垂；若對準誤差 > 容許值，可退回此階段最多修正 2 次 |
| **4** | 開爪釋放＋退避 | 張爪釋放方塊後垂直退避至安全高度，消除放置後的夾爪擠壓回彈與方塊擺動 |

---

## 🛠️ 環境需求

| 項目 | 規格 |
|------|------|
| 模擬平台 | **NVIDIA Isaac Sim 6.0** |
| Python | **3.10+**（由 Isaac Sim 內建環境提供） |
| GPU | **NVIDIA GPU**（需支援 RTX 光追，建議 RTX 3070 以上） |
| 作業系統 | Ubuntu 22.04 LTS（官方支援） |

---

## 🚀 快速啟動

```bash
# 3 層河內塔（7 步，適合初學驗證）
./run.sh --disks 3

# 5 層河內塔（31 步，觀察避障走廊動態調整）
./run.sh --disks 5

# 8 層河內塔（255 步，長時間耐久測試）
./run.sh --disks 8

# 10 層河內塔（1023 步，最高挑戰）
./run.sh --disks 10

# 選擇性以 20 Hz 輸出末端世界位置與四元數方向（預設 false）
./run.sh --disks 3 --log-end-effector-pose true
```

> **支援層數範圍**：`--disks` 參數可接受 2 至 10（對應 3 至 1023 步）。
>
> 任務完成或中止時，pandas 移動表會印在終端並存成 CSV，與 MP4 使用相同檔名主體。
> 例如影片 `recordings/hanoi_3disks_時間戳.mp4` 對應表格
> `recordings/hanoi_3disks_時間戳.csv`。預設不輸出末端姿態；使用
> `--log-end-effector-pose true` 才會開啟 20 Hz 診斷列印。

---

## 📁 資料夾結構

```
hanoi_tower/
├── main.py                     # 模擬主程式入口（參數解析、視窗、相機視角、自動 MP4 錄影）
├── run.sh                      # 執行腳本（自動定位本機 Isaac Sim python.sh）
├── README.md                   # 專案說明文件（本文件）
├── .gitignore                  # Git 忽略清單（排除大型影片、虛擬環境與快取）
├── hanoi_tower/                # 核心控制套件
│   ├── __init__.py             # 套件模組初始化
│   ├── planner.py              # 純演算法模組：河內塔分治求解、S 曲線、高度計算
│   ├── franka.py               # Franka 機器人介面：雅可比 dIK、阻尼自適應、零空間投影
│   ├── pick_place.py           # 5 階段取放有限狀態機（FSM）與 S 曲線軌跡插值
│   └── tower.py                # 河內塔場景建置、方塊階梯寬度、避障巡航走廊與貼地校正
├── tests/                      # 單元測試套件
│   └── test_planner.py         # 測試河內塔步數最優性、合法性、平滑步進函數
└── recordings/                 # 錄影影片儲存目錄
    └── .gitkeep                # 保留目錄結構
```

---

## 📤 GitHub 分支工作流程

```bash
# 初始化並推送目前專案
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/nekoluga748/Tower-of-Hanoi.git

# 建立整合分支；各功能分支由 dev 分出
git switch -c dev
git push -u origin dev
git switch -c feature/planner dev
# 在此分支修改並提交 Hanoi 規劃器
git switch dev
git merge --no-ff feature/planner
git switch -c feature/franka-control
# 在此分支修改並提交 Franka 與取放控制
git switch dev
git merge --no-ff feature/franka-control
git switch -c feature/tower-scene
# 在此分支修改並提交場景與堆疊管理
git switch dev
git merge --no-ff feature/tower-scene
git switch -c feature/app-entry
# 在此分支修改並提交應用程式入口

# 功能分支完成後合併回 dev，再發布到 main
git switch dev
git merge --no-ff feature/app-entry
git push origin dev
git switch main
git merge --no-ff dev
git push origin main
```

> `.gitignore` 會排除 `.venv/` 與 `recordings/*.mp4` 等大型或本機產生的檔案。
> `examples/original_franka/` 是未修改上游範例的存放位置；目前工作區未附上該原始
> 範例，因此該資料夾只保留來源與授權說明。
---

## ❓ 常見問題 Q&A

**Q1：執行 `./run.sh` 時出現「找不到 Isaac Sim」錯誤？**
> `run.sh` 會先檢查 `~/isaacsim`，再搜尋 `~/.local/share/ov/pkg/isaac-sim*`。若安裝於其他位置，執行前設定 `ISAAC_SIM_PATH=/你的/Isaac-Sim-安裝目錄` 即可，不必修改腳本。

**Q2：單元測試可以在沒有 GPU 的環境下執行嗎？**
> 可以。`tests/test_planner.py` 僅測試純 Python 演算法模組（河內塔步驟生成、S 曲線插值、高度計算），不依賴 Isaac Sim 或 GPU，可在任何具備 Python 3.10+ 的環境中以 `python3 -m unittest discover -s tests -v` 秒級通過。

**Q3：增加盤子數量時，模擬速度變慢的原因？**
> 步驟數以 $2^N - 1$ 指數成長（10 層需執行 1,023 次完整 Pick-Place 循環），每個循環包含 5 個 Phase，總幀數極大。建議搭配 `--headless` 無圖形介面模式以提升模擬速度，或縮短每 Phase 的步進時間（調整 `events_dt` 參數）。

---

## 📚 NVIDIA 外部資源與程式來源

- **NVIDIA Isaac Sim 6.0：** 執行平台。`SimulationApp`、Timeline、Extension Manager、`SimulationManager`、USD Stage 及實驗性 `Cube`、`RigidPrim`、`GeomPrim`、`RigidBodyMaterial` 等 API 由 Isaac Sim 提供。
- **NVIDIA Franka Panda USD：** `franka.py` 經 `get_assets_root_path()` 載入 `/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`，選用 `AlternateFinger` 夾爪及 `Performance` 網格變體。機器人模型是 NVIDIA 資產，不是本專案自製。
- **NVIDIA Franka 抓取／放置範例脈絡：** `franka.py`、`pick_place.py` 保留 NVIDIA Copyright 與 Apache-2.0 SPDX 標頭，承接官方 Franka articulation 及單物件抓放操作模式；本專案在其上加入 Hanoi 遞迴任務層、多盤三柱場景、符號堆疊同步、抬升檢查及放置校正。重散布程式時應遵守這兩個檔案中的 Apache-2.0 授權條款。
- **NVIDIA Warp 與 Isaac Sim 四元數工具：** `franka.py` 用於四元數乘法及共軛運算；Jacobian 誤差解算流程與 Hanoi 遞迴是本專案程式，不應誤標為 Warp 或 NVIDIA Hanoi 演算法。
- **NVIDIA Synthetic Data：** 錄影模式下擷取視口 RGB；OpenCV 負責 MP4 編碼，pandas 負責移動表格與 CSV，兩者為第三方 Python 套件。
- **參考文件：** [NVIDIA Isaac Sim Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/)。外部 USD 資產實際路徑以 `franka.py` 的引用字串為準。

**本專案自行實作：** `planner.py` 的 Hanoi 遞迴與數值輔助函式；`tower.py` 的三柱盤位模型、物理位姿校正和失敗重試；`pick_place.py` 的本專案五階段時序與位置插值；`franka.py` 的 dIK 求解控制流程；以及 `main.py` 的 pandas 移動紀錄。NVIDIA 提供模擬器、機器人 USD、底層物理／Articulation 能力與相關 API，不是這些 Hanoi 任務演算法的作者。
