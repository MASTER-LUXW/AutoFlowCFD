# AutoFlowCFD 教程

用仓库自带的网格与验证算例走一遍完整流程。命令都可以原样运行；文中给出的数值是记录在案的实测结果（注明网格、
阶数与模型），不是期望值或文献值。命令选项以 `--help` 为准，配置字段见 [配置指南](CONFIGURATION_GUIDE.md)。

---

## 目录

- [教程 1：立方体绕流稳态 RANS](#教程-1立方体绕流稳态-rans)
- [教程 2：续算、升阶与换格式](#教程-2续算升阶与换格式)
- [教程 3：瞬态 DDES、探针与压力谱](#教程-3瞬态-ddes探针与压力谱)
- [教程 4：GPU 与分布式](#教程-4gpu-与分布式)
- [教程 5：Python API 批量计算](#教程-5python-api-批量计算)
- [教程 6：Ahmed 体](#教程-6ahmed-体)
- [验证算例](#验证算例)
- [常见问题](#常见问题)

---

## 教程 1：立方体绕流稳态 RANS

`examples/cube_demo/cube_demo.nas`：带圆角的立方体（边长 1 m）放在风洞里，4 个边界组 inlet / outlet / tunnel / body，
坐标单位米。

```bash
# 1. 体网格（边界层棱柱 + 四面体），打印质量报告
poetry run autoflowcfd grid generate-volume examples/cube_demo/cube_demo.nas --units m -o cube/cube_volume.nas

# 2. 稳态 SA，P0 -> P1 逐阶爬坡，隐式 Newton-Krylov
poetry run autoflowcfd solve steady cube/cube_volume.nas --surface-mesh examples/cube_demo/cube_demo.nas --units m \
    --order 1 --turbulence-model sa --time-scheme newton-krylov --max-iter 300 --checkpoint-interval 20 -o cube/steady
```

每步日志一行：残差、相对本阶段起点的下降倍数、CFL、Newton 诊断（GMRES 次数、步长）、压力积分力系数
`Cd_p / Cl_p / Cs_p`；第 1 步与之后每 10 步另附逐方程缩放残差、最大残差所在单元，以及累计伪时间相对流动特征时间的比例。
残差下降到 `--tol`（缺省 1e-6）倍即收敛。

**实测**（37.4 万单元、来流 33.33 m/s、SA、P1）：残差比 P1 起点下降约 1e6 倍，`Cd ≈ 0.89`、`Cl`、`Cs` 接近 0。这个
立方体带圆角，阻力明显低于尖棱立方体（约 1.05）。剩余残差主要在物体侧面分离区的 nu_tilde 方程（缓慢的涡粘模态）。

后处理：

```bash
poetry run autoflowcfd post coefficients --case cube/steady --reference-area 1.0 --reference-length 1.0 \
    --moment-center 0.5 0 0 --groups body
poetry run autoflowcfd post export-vtk --case cube/steady -o cube/steady/flow.vtu \
    --variables velocity --variables pressure --variables q_criterion
poetry run autoflowcfd post export-vtk --case cube/steady --boundaries-only -o cube/steady/walls.vtu
poetry run autoflowcfd post convergence --case cube/steady --variables residual --variables coefficients -o conv.png
```

ParaView 打开 `flow.vtu`：`Pressure` 着色看前缘滞止区与侧面负压区；`Velocity` 做流线看尾流回流区；`QCriterion`
取正的等值面看涡结构。`walls.vtu` 带 `BoundaryID` 与组名图例，可按组筛选。

---

## 教程 2：续算、升阶与换格式

**续算**：物理参数（来流、粘度、湍流模型、近壁处理、合成湍流入口涡核数……）一律取自 checkpoint，缺失即报错；CFL 等
纯数值参数每次可以重新给。

```bash
poetry run autoflowcfd solve resume cube/steady/checkpoints/checkpoint_iter_000300.h5 --max-iter 200 --cfl-start 3000
```

**换稳态格式**：`--time-scheme rk3 / newton-krylov` 在两个稳态格式之间切换（时间精确的瞬态算例不能换）。

**升阶**：`solve resume` 不改目标阶数；要在收敛的 P1 解上直接算 P2，用 `--init-from`（解与湍流场精确延拓到新阶数，
不重走爬坡）：

```bash
poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --order 2 --turbulence-model sa \
    --time-scheme newton-krylov --init-from cube/steady/checkpoints/checkpoint_iter_000300.h5 -o cube/steady_p2
```

内存随阶数快速增长：37.4 万单元的 SA P2 隐式需要 45 GB 以上；内存不够时用分布式（教程 4）分摊。

---

## 教程 3：瞬态 DDES、探针与压力谱

从稳态解起步做 SST-DDES，速度入口自动注入合成湍流：

```bash
poetry run autoflowcfd solve transient cube/steady/volume_mesh.pkl --order 1 --turbulence-model ddes \
    --time-method dual-time --dt 1e-3 --physical-time 0.5 --checkpoint-interval 10 \
    --init-from cube/steady/checkpoints/checkpoint_iter_000300.h5 \
    --probe 1.5 0.0 0.0 --probe 2.5 0.3 0.0 -o cube/ddes
```

- `dual-time`：每个物理步若干个带 BDF2 物理时间项的隐式 Newton 步（`--dual-time-inner-iter` 为上限，残差降到物理步
  起点的 1e-3 即提前结束）。实测立方体 SA P1 从收敛解起步、dt = 1e-3：每步 2 次内迭代。
- `rk3`：显式、全场同一物理步长的时间精确推进（每个 `--dt` 切成若干显式稳定的子步）。
- `--probe`：每个物理步在探针处求 FR 解多项式的值，写进 `cube/ddes/probes.csv`（time、iteration、各探针 p u v w）；续算
  从 checkpoint 时刻接着写。

统计与频谱（`--warmup-time` 排除起动瞬态，按 checkpoint 记录的物理时间）：

```bash
poetry run autoflowcfd post transient-mean --case cube/ddes --warmup-time 0.1        # MeanVelocity / MeanPressure + Cd/Cl/Cs 时均
poetry run autoflowcfd post transient-rms --case cube/ddes --warmup-time 0.1         # VelocityRMS / PressureRMS
poetry run autoflowcfd post transient-psd --case cube/ddes --warmup-time 0.1 -o psd.csv   # 探针压力的 Welch 谱
```

时均与 RMS 的样本是各个 checkpoint，样本间隔即 `--checkpoint-interval` 个物理步；压力谱的样本是每个物理步的探针值。

---

## 教程 4：GPU 与分布式

```bash
# 单 GPU
poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --backend gpu --order 1 --turbulence-model sa

# CPU MPI：每个 rank 加载完整网格（传统模式）；只让 root 加载、其余 rank 只收切好的紧凑数据：加 --fully-distributed
mpirun -np 8 poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --n-ranks 8 --order 1 --turbulence-model sa

# 多 GPU：每个 rank 一块 GPU（不给 --gpu-device 时按 rank % GPU 数分配）
mpirun -np 4 poetry run autoflowcfd solve steady cube/steady/volume_mesh.pkl --backend gpu --multi-gpu --n-ranks 4
```

四个后端是同一套算法、同一个求解循环、同一份 checkpoint 格式：分布式的 checkpoint 可以单机续算，反之亦然
（`solve resume ... --n-ranks N`）。BLAS 线程数缺省为"本节点核数 / 本节点 rank 数"，numba 线程数由 `--threads` 给出。

---

## 教程 5：Python API 批量计算

参数名就是配置字段；每个算例写到自己的输出目录，checkpoint 可用 `solve resume` 续算。

```python
from autoflowcfd import AutoFlowCFDAPI

api = AutoFlowCFDAPI()
volume_mesh = api.generate_volume_mesh(api.load_grid("examples/cube_demo/cube_demo.nas", units="m"))

rows = []
for aoa in (0.0, 5.0, 10.0):
    result = api.run_steady(volume_mesh, order=1, turbulence="sa", time_scheme="newton-krylov", max_iter=300,
                            output_dir=f"./sweep/aoa_{aoa:g}", aoa_deg=aoa)
    c = api.calculate_coefficients(reference_area=1.0)
    rows.append((aoa, result.converged, c["Cd"], c["Cl"]))

for aoa, ok, cd, cl in rows:
    print(f"aoa={aoa:4.1f}  converged={ok}  Cd={cd:.4f}  Cl={cl:.4f}")
```

`aoa_deg` / `aos_deg` 是来流方向（攻角绕 y 轴、抬头为正；侧滑角绕 z 轴），来流边界、初场、合成湍流入口、气动力的风轴系
分解与自动参考面积都按它构造。

---

## 教程 6：Ahmed 体

`examples/ahmed_demo/ahmed_body_demo.nas`：Ahmed 体 + 地面 + 风洞（组 body / ground / inlet / outlet / tunnel，毫米），
`config_example.yaml` 是对应的配置文件：

```bash
poetry run autoflowcfd grid generate-volume examples/ahmed_demo/ahmed_body_demo.nas -o ahmed/ahmed_volume.nas \
    --target-yplus 30 --ref-velocity 40 --ref-length 1.044
poetry run autoflowcfd solve steady ahmed/ahmed_volume.nas --surface-mesh examples/ahmed_demo/ahmed_body_demo.nas \
    --config examples/ahmed_demo/config_example.yaml
```

地面按固定壁面处理（不支持移动地面）。这个算例目前只演示流程，还没有与 Ahmed 实验数据做过对比验证。

---

## 验证算例

`tests/validation/` 下的算例有精确解、关联式或守恒量可以定量核对（`poetry run pytest tests/validation -q -o addopts=`）：

| 算例 | 核对的量 | 实测 |
|------|---------|------|
| Blasius 层流平板（`test_blasius.py`） | 壁面摩阻与速度剖面对 Blasius 精确解 | — |
| 平面 Couette（`test_couette.py`） | 线性速度剖面 | — |
| 湍流平板近壁处理（`test_flat_plate_wall_treatment.py`） | 局部摩阻对 White 关联式（0.5 < x < 1.8，P1 SST） | 第一层 2e-5 / 2e-4 / 2e-3 m（形心 y+ 约 1.8 / 19 / 190）：`auto` -6.4% / +2.0% / +1.8%，`resolved` -13.2% / -77.5% / -97.3% |
| 等熵涡（`test_isentropic_vortex.py`） | 涡平动后的位置与形状 | — |
| 双时间步时间精度（`test_dual_time_temporal_order.py`） | 等熵涡 P2，dt 减半误差比 | 时间阶数 2.05 / 2.10 |
| Taylor-Green 涡（`test_tgv.py`） | 粘性动能衰减、周期域质量 / 能量守恒 | — |
| 周期边界（`test_periodic_bc.py`） | 周期面配对与自一致性 | — |
| 贴壁层涡粘（`test_wall_layer_eddy_viscosity.py`） | 壁面外插负涡粘不向贴壁层注入动量 | 贴壁层最大速度 30.8 m/s（来流 30 m/s） |

"—" 表示判据写在测试里（通过即满足），没有单独记录的数值。

---

## 常见问题

**不收敛** —— 看每步日志里最大残差所在单元：在畸形单元上时先改网格；显式格式调低 `--cfl-start`；复杂几何用
`--time-scheme newton-krylov`；先在低阶收敛再 `--init-from` 升阶。

**选哪个湍流模型** —— 稳态 RANS：SA（高阶隐式收敛最好）或 SST；分离与尾流的非定常：DDES / IDDES；壁面解析 LES 网格
代价高，壁面附近网格不够时用 WMLES。近壁网格第一层 y+ 不在 1 附近时保持 `--wall-treatment auto`。

**显存 / 内存不够** —— 降阶、用 `--n-ranks` 分到多个进程或多台机器，或用多 GPU。

---

下一步：[后处理指南](POSTPROCESSING_GUIDE.md)、[API 文档](API.md)、[配置指南](CONFIGURATION_GUIDE.md)。
