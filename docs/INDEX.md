# 文档索引

## 使用

| 文档 | 内容 |
|------|------|
| [README](../README.md) | 功能概览、安装、源码结构 |
| [QUICKSTART](../QUICKSTART.md) | 安装与第一个算例（面网格 -> 体网格 -> 求解 -> 后处理） |
| [INSTALL](../INSTALL.md) | 依赖、可选组（gpu / mpi / perf）、环境检查 |
| [教程](TUTORIALS.md) | 立方体稳态 RANS、续算与升阶、瞬态 DDES 与探针、GPU / 分布式、API 批量计算、验证算例 |
| [配置指南](CONFIGURATION_GUIDE.md) | 命令行 / YAML / Python 三种给法与优先级，全部配置字段 |
| [边界条件](boundary_configuration_guide.md) | 边界类型、按组名识别、`--bc-type` 显式指定 |
| [后处理指南](POSTPROCESSING_GUIDE.md) | 气动系数、VTK、收敛曲线与报告、瞬态统计与压力谱、Python 接口 |
| [VTK 导出指南](VTK_EXPORT_GUIDE.md) | 低阶 / 高阶导出、边界面片、网格转换、ParaView 用法 |
| [API 参考](API.md) | `AutoFlowCFDAPI` 的全部方法 |
| [性能与资源](PERFORMANCE_OPTIMIZATION.md) | 计时、格式选择、线程、内存估计 |
| [`.pkl` 体网格](PKL_GRID_FORMAT.md) | 求解器原生体网格文件的来源与内容 |

每条命令的完整选项：`autoflowcfd <命令组> <命令> --help`。

## 开发

| 文档 | 内容 |
|------|------|
| [开发者指南](DEVELOPER_GUIDE.md) | 代码组织、约定、测试、扩展点 |
| [架构](../ARCHITECTURE.md) | 模块职责与数据流 |
| [core 模块说明](../src/autoflowcfd/core/README.md) | 求解器核心的入口与用法 |
| [grid 模块说明](../src/autoflowcfd/grid/README.md) | 网格子系统 |
| [贡献指南](../CONTRIBUTING.md) | 流程与代码规范 |

算法细节以源码模块文档为准（各子包 `__init__.py` 与关键模块开头的文档字符串，例如
`core/time_integration/adaptive_cfl`、`core/turbulence/transported.py`、`core/fr_solver/solver/solve_loop.py`）。

## 项目

[CHANGELOG](../CHANGELOG.md) · [ROADMAP](../ROADMAP.md) · [SECURITY](../SECURITY.md) · [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md)

问题反馈：[GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
