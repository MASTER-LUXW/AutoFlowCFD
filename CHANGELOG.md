# 变更日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/) (SemVer) 规范。

## [Unreleased]

### Added
- **Iteration 5: Postprocessing & Integration** - 完整的后处理模块和集成测试
  - 气动系数计算模块（CoefficientCalculator）
    - Cd/Cl/Cm/Cs/Cy/Cr六分量系数计算
    - 力和力矩绝对值计算
    - 分边界系数计算框架
  - VTK场数据导出模块（VTKExporter）
    - Legacy VTK格式支持（ASCII）
    - 速度、压力、湍流变量导出
    - ParaView兼容格式
  - 收敛分析与报告生成（ConvergenceAnalyzer, SimulationReport）
    - CSV收敛曲线导出
    - JSON仿真报告生成
    - 收敛判定逻辑
  - 瞬态统计后处理（TransientStatistics, PressurePSD）
    - 时均流场计算（Welford在线算法）
    - RMS脉动量计算
    - FFT功率谱密度分析
    - 主导频率识别
  - Ahmed Body Demo算例
    - 稳态RANS配置和运行脚本
    - 瞬态DES配置和运行脚本
  - 性能基准测试套件
    - CPU基准测试（稳态RANS + 瞬态DES）
    - GPU基准测试（稳态RANS + 瞬态DES/LES）
  - SolutionVector数据结构（core.backend.base）
  - 34个单元测试用例（test_postprocess.py）
  - 快速验证脚本（scripts/verify_iteration5.py）

### Changed
- 更新backend模块导出SolutionVector类
- 完善postprocess模块__init__.py导出所有公共类

### Deprecated
- 无

### Removed
- 无

### Fixed
- 添加SolutionVector数据类定义（之前缺失）
- 修复后处理模块导入路径

### Security
- 所有数值计算添加输入验证
- 文件路径安全检查

---

## [0.1.0] - 2026-07-23

### Added
- **工程基础设施**（Iteration 1）
  - Git仓库初始化与分支策略配置
  - 项目目录结构搭建
  - Poetry依赖管理配置
  - CI/CD流水线（GitHub Actions）
  - Pre-commit钩子配置
  - 基础模块占位实现

### 计划中功能
- **网格解析模块**（Iteration 2，第3-4周）
  - NAS文件解析器（v22/v23/v24格式支持）
  - 网格数据结构（SoA内存布局）
  - 网格质量校验器
  - 边界条件识别与映射
  
- **FR求解器核心**（Iteration 3，第5-8周）
  - FR离散格式实现
  - SST k-ω湍流模型
  - DES/DDES湍流模型
  - CPU后端（Numba并行）
  - GPU后端（CUDA加速）
  
- **CLI与API接口**（Iteration 4，第8-9周）
  - Click命令行界面
  - Python API封装
  - JSON输出格式
  
- **后处理与集成**（Iteration 5，第10-11周）
  - 气动系数计算
  - VTK场数据导出
  - 收敛曲线分析
  - 端到端测试

---

## 版本说明

### 版本号格式

`主版本号.次版本号.修订号`

- **主版本号**：不兼容的API修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

### 变更类型说明

- **Added**：新功能
- **Changed**：现有功能的变更
- **Deprecated**：即将移除的功能
- **Removed**：已移除的功能
- **Fixed**：Bug修复
- **Security**：安全相关修复或改进

---

## 链接

- [GitHub Releases](https://github.com/AutoFlowCFD/AutoFlowCFD/releases)
- [Issue Tracker](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- [Project Roadmap](ROADMAP.md)

[Unreleased]: https://github.com/AutoFlowCFD/AutoFlowCFD/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/AutoFlowCFD/AutoFlowCFD/releases/tag/v0.1.0
