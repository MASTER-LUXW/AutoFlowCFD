# 路线图

## 现状（0.1.0，2026-10）

已具备（详见 [README](README.md)）：FR P0~P3（原生四面体 / 三棱柱基）；AUSM+up + BR1；SST、SA-neg、DDES / IDDES、
LES（WALE）、WMLES，RANS 自动近壁处理；显式 / 隐式 Newton-Krylov 稳态，双时间步与显式时间精确瞬态；CPU、单 GPU、
CPU-MPI、多 GPU 四个后端同一算法；体网格生成与导入；气动系数、高阶 VTK、瞬态统计、求解器侧探针与压力谱；
checkpoint 续算。

## 近期

- **验证**：Ahmed 体（25° / 35°）与 DrivAer 的阻力、升力、表面压力对实验数据；网格与阶数收敛性。目前定量验证只有
  `tests/validation` 的标准算例（Blasius、Couette、湍流平板、等熵涡、TGV）。
- **全仓库评审遗留项**：按 `ProjectFiles` 里的评审台账逐项处理（剩余为 C/D 级的结构与文档问题）。
- **P2 / P3 在工程网格上的收敛与内存**：隐式预处理内存、平均流与湍流之间的耦合预处理。

## 中期

- 运动壁面：移动地面、旋转车轮（幽灵态已支持壁面速度，缺用户入口与验证）。
- 曲边高阶单元（目前为直边单元）。
- 网格自适应加密。
- 多 GPU 之间的直接通信（目前 halo 经主机中转或 CUDA-aware MPI）。

## 长期

- FW-H 声类比（远场气动噪声）。
- 伴随灵敏度分析。
- 传热等多物理场耦合。

计划会随验证结果调整；有需求或想参与的方向请在 [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues) 讨论。
