# 边界条件

边界条件类型是物理输入，来自网格的边界分组：面网格 `.nas` 的每个 `PSHELL` 属性是一个组，组名决定类型（或由用户显式
指定）。实现：类型识别 `grid/nas_io/boundary_types.py`，求解器幽灵态 `core/fr_solver/boundary/ghost.py` 与
`boundary/fr_ghost_state.py`。

---

## 类型

| 类型 | 求解器处理 | 典型用途 |
|------|-----------|---------|
| `WALL` | 无滑移固壁（绝热）；RANS 按 `--wall-treatment` 处理近壁，WMLES 由壁面模型给剪应力；积分进气动力 | 车身、车轮、地面 |
| `SLIP_WALL` | 无摩擦壁（法向速度为零、切向自由滑移）；不挤出边界层，不计入气动力 | 风洞洞壁 |
| `VELOCITY_INLET` | 入口给定来流状态（`--vel-inf --rho-inf --p-inf`、方向按 `--aoa --aos`）；LES/WMLES/DDES/IDDES 叠加合成湍流（SEM） | 入口 |
| `PRESSURE_OUTLET` | 出口给定静压 `--p-inf` | 出口 |
| `SYMMETRY` | 对称面 | 半模 |
| `FARFIELD` | 远场（来流特征边界） | 外流开放边界 |
| `PERIODIC` | 周期面配对后成为内部面（只能在加载网格前经 Python 给出配对组与平移向量） | 验证算例 |

移动地面、旋转车轮等运动壁面目前没有用户入口：地面按固定 `WALL` 处理。

---

## 由组名识别

组名按非字母数字字符切成词，英文关键字整词匹配（`sym` 不会命中 `asymmetric`），中文关键字子串匹配：

| 类型 | 英文关键字 | 中文关键字 |
|------|-----------|-----------|
| `VELOCITY_INLET` | inlet、inflow、entrance | 入口 |
| `PRESSURE_OUTLET` | outlet、outflow、exit | 出口 |
| `SYMMETRY` | symmetry、sym、symm | 对称 |
| `FARFIELD` | farfield、freestream、far | 远场 |
| `SLIP_WALL` | slip、tunnel | 风洞、洞壁 |
| `WALL` | wall、body、ground、floor | 车体、车身、壁面、地面 |

没有命中任何关键字、或同时命中两种类型（如 `TUNNEL_WALL`）的组类型未定，体网格生成与导入会报错并列出这些组，必须显式
指定。

## 显式指定

```bash
autoflowcfd grid generate-volume car.nas -o car_volume.nas --bc-type TUNNEL_WALL=SLIP_WALL --bc-type MIRROR=SYMMETRY
autoflowcfd grid import-volume ext_volume.nas --surface-mesh car.nas -o car_volume.pkl --bc-type WHEELS=WALL
```

`--bc-type 组名=类型` 可重复给出，覆盖按名字识别的结果；组名必须存在、类型必须是上表（除 `PERIODIC`）之一。类型保存在
`.pkl` 体网格里，求解、续算与后处理沿用。`.nas` 文件不记录类型，重新读入时按组名识别，所以求解命令也接受
`--bc-type`：输入是 `.nas` 体网格时在导入时应用、写进输出目录的 `volume_mesh.pkl`；输入是 `.pkl` 时把覆盖后的副本写到
输出目录（原文件不改）：

```bash
autoflowcfd solve steady car_volume.nas --surface-mesh car.nas --bc-type TUNNEL_WALL=SLIP_WALL -o results
```

`grid info` 列出各组及单元数；`grid convert -f vtk` 导出的网格带组号，可在 ParaView 里核对分组。

---

## 求解时的检查

- 每个边界面必须属于某个组，每个组必须有类型；缺任何一个直接报错（不会把没有组的面当成远场）。
- 速度入口与出口的来流参数取自求解命令（或续算时的 checkpoint），没有逐组参数。
- 求解结束时打印壁面 y+ 统计；第一层 y+ 超出所选近壁处理的适用范围时警告。

Python 层可用 `FRSolver(..., bc_overrides={组名: 幽灵态配置})` 直接替换某组的幽灵态配置（测试与验证算例使用，
配置字典的键见 `core/fr_solver/boundary/ghost.py` 的类型表）；命令行不提供这一入口。
