# AutoFlowCFD 文档中心

欢迎来到 AutoFlowCFD 文档中心！这里提供了从入门到高级的完整文档资源。

---

## 📚 文档导航

### 🚀 快速开始

| 文档 | 说明 | 适合人群 | 阅读时间 |
|------|------|---------|---------|
| [README](../README.md) | 项目概述与核心特性 | 所有人 | 5 分钟 |
| [QUICKSTART](../QUICKSTART.md) | 5 分钟完成安装与首次仿真 | 新用户 | 10 分钟 |
| [INSTALL](../INSTALL.md) | 详细安装指南 | 新用户 | 15 分钟 |

---

### 📖 用户文档

#### 基础教程

| 文档 | 说明 | 难度 |
|------|------|------|
| [算例教程](TUTORIALS.md) | 完整的 CFD 仿真算例，从简单到复杂 | ⭐⭐ |
| [配置指南](CONFIGURATION_GUIDE.md) | YAML 配置文件详解与最佳实践 | ⭐⭐ |
| [后处理指南](POSTPROCESSING_GUIDE.md) | 结果可视化与数据分析 | ⭐⭐ |

#### 进阶主题

| 文档 | 说明 | 难度 |
|------|------|------|
| [API 参考](API.md) | Python API 完整参考文档 | ⭐⭐⭐ |
| [性能优化](PERFORMANCE_OPTIMIZATION.md) | 最大化计算效率的策略 | ⭐⭐⭐ |
| [VTK 导出指南](VTK_EXPORT_GUIDE.md) | 可视化数据导出详解 | ⭐⭐ |
| [边界条件配置](boundary_configuration_guide.md) | 边界条件系统详细说明 | ⭐⭐⭐ |

---

### 💻 开发者文档

| 文档 | 说明 | 适合人群 |
|------|------|---------|
| [开发者指南](DEVELOPER_GUIDE.md) | 二次开发、代码规范、测试流程 | 贡献者 |
| [架构设计](../ARCHITECTURE.md) | 系统架构与模块划分 | 开发者 |
| [编码规范](../ProjectFiles/2-7_编码规范-Part1.md) | 代码风格与质量标准 | 贡献者 |
| [接口文档](../ProjectFiles/2-4_接口文档-Part1.md) | API/CLI 接口规范 | 开发者 |

---

### 🔧 技术文档

#### 核心算法

| 文档 | 说明 |
|------|------|
| [FR 离散格式](../ProjectFiles/3-4_系统实现方式-Part2-Core.md) | 通量重构算法实现细节 |
| [湍流模型](../src/autoflowcfd/core/README.md) | SST k-ω、DES/DDES 模型说明 |
| [时间积分方案](../src/autoflowcfd/core/time_integration.py) | BE/RK2/AB3 时间离散 |

#### 数据结构

| 文档 | 说明 |
|------|------|
| [数据结构设计](../ProjectFiles/2-3_数据结构设计文档-Part1.md) | SoA 布局与内存优化 |
| [网格系统](../ProjectFiles/3-4_系统实现方式-Part1-Grid.md) | NAS 解析与网格处理 |
| [PKL 网格格式](PKL_GRID_FORMAT.md) | 二进制网格缓存格式 |

#### 性能与优化

| 文档 | 说明 |
|------|------|
| [CFL 自适应优化](CFL_ADAPTIVE_OPTIMIZATION.md) | 自适应 CFL 策略详解 |
| [日志格式优化](LOG_FORMAT_OPTIMIZATION.md) | 结构化日志系统 |
| [VTK 数据基础](VTK_DATA_BASIS.md) | VTK 文件格式与导出原理 |

---

### 🛠️ 工具与脚本

| 文档 | 说明 |
|------|------|
| [VTK CLI 使用指南](VTK_CLI_GUIDE.md) | VTK 命令行工具使用 |
| [VTK 故障排查](VTK_TROUBLESHOOTING.md) | VTK 导出常见问题解决 |
| [VTK 快速参考](VTK_QUICK_REFERENCE.md) | VTK 导出速查表 |
| [开发工具使用指南](开发工具使用指南.md) | Poetry、pytest、Black 等工具 |

---

### 📋 项目文档

#### 规划与设计

| 文档 | 说明 |
|------|------|
| [项目实施路径](../ProjectFiles/0_项目实施路径.md) | 项目生命周期管理 |
| [立项说明书](../ProjectFiles/1-1_立项说明书.md) | 项目背景与定位 |
| [竞品分析](../ProjectFiles/1-2_竞品分析文档.md) | 市场分析与差异化 |
| [系统架构](../ProjectFiles/2-2_系统架构文档-Part1.md) | 整体架构设计 |

#### 开发与迭代

| 文档 | 说明 |
|------|------|
| [迭代开发计划](../ProjectFiles/3-1_迭代开发计划-Part1.md) | Roadmap 与里程碑 |
| [版本管理验收](../ProjectFiles/3-2_开发与版本管理验收-Part1.md) | 版本发布流程 |
| [重要问题优化](../ProjectFiles/3-3_开发阶段重要问题优化-Part1.md) | 技术难点攻关记录 |

---

### 🤝 社区与贡献

| 文档 | 说明 |
|------|------|
| [CONTRIBUTING](../CONTRIBUTING.md) | 贡献指南 |
| [CODE_OF_CONDUCT](../CODE_OF_CONDUCT.md) | 社区行为准则 |
| [SECURITY](../SECURITY.md) | 安全策略 |
| [ROADMAP](../ROADMAP.md) | 项目路线图 |
| [CHANGELOG](../CHANGELOG.md) | 版本变更日志 |

---

## 🎯 学习路径推荐

### 新手入门（第 1 周）

1. ✅ 阅读 [README](../README.md) 了解项目
2. ✅ 跟随 [QUICKSTART](../QUICKSTART.md) 完成首次仿真
3. ✅ 运行 [立方体绕流算例](TUTORIALS.md#算例-1-立方体绕流验证)
4. ✅ 学习 [配置基础](CONFIGURATION_GUIDE.md#基础配置)

**目标**: 能够独立运行稳态 RANS 仿真

---

### 进阶用户（第 2-4 周）

1. ✅ 完成 [Ahmed Body 标准算例](TUTORIALS.md#算例-2-ahmed-body-标准算例)
2. ✅ 学习 [后处理与可视化](POSTPROCESSING_GUIDE.md)
3. ✅ 尝试 [瞬态 DDES 仿真](TUTORIALS.md#算例-4-瞬态尾流分析)
4. ✅ 阅读 [性能优化指南](PERFORMANCE_OPTIMIZATION.md)

**目标**: 能够独立完成工程级汽车外流场仿真

---

### 高级用户（第 2-3 月）

1. ✅ 学习 [API 高级用法](API.md#高级用法)
2. ✅ 进行 [参数化优化研究](TUTORIALS.md#算例-5-参数化优化研究)
3. ✅ 自定义 [边界条件扩展](DEVELOPER_GUIDE.md#新增边界条件)
4. ✅ 贡献代码或文档

**目标**: 能够扩展功能并贡献社区

---

### 开发者（持续）

1. ✅ 阅读 [开发者指南](DEVELOPER_GUIDE.md)
2. ✅ 理解 [系统架构](../ARCHITECTURE.md)
3. ✅ 遵循 [编码规范](../ProjectFiles/2-7_编码规范-Part1.md)
4. ✅ 参与 Code Review 和 Issue 讨论

**目标**: 成为核心贡献者

---

## 🔍 快速查找

### 常见问题

| 问题 | 参考文档 |
|------|---------|
| 如何安装？ | [QUICKSTART](../QUICKSTART.md#安装步骤) |
| 如何运行仿真？ | [QUICKSTART](../QUICKSTART.md#首次运行) |
| 如何配置求解器？ | [配置指南](CONFIGURATION_GUIDE.md#求解器配置) |
| 如何选择湍流模型？ | [配置指南](CONFIGURATION_GUIDE.md#湍流模型选择) |
| 如何可视化结果？ | [后处理指南](POSTPROCESSING_GUIDE.md) |
| 如何提高性能？ | [性能优化](PERFORMANCE_OPTIMIZATION.md) |
| 如何添加新功能？ | [开发者指南](DEVELOPER_GUIDE.md#扩展开发) |
| 如何提交 PR？ | [CONTRIBUTING](../CONTRIBUTING.md) |

### 按主题查找

#### 网格相关
- NAS 文件解析: [网格系统](../ProjectFiles/3-4_系统实现方式-Part1-Grid.md)
- 网格质量校验: [QUICKSTART](../QUICKSTART.md#网格准备)
- 网格格式转换: [VTK CLI 指南](VTK_CLI_GUIDE.md)

#### 求解器相关
- FR 离散格式: [核心算法](../ProjectFiles/3-4_系统实现方式-Part2-Core.md)
- 湍流模型: [湍流模型说明](../src/autoflowcfd/core/README.md)
- 收敛控制: [配置指南](CONFIGURATION_GUIDE.md#求解器配置)

#### 边界条件
- 边界类型: [边界配置指南](boundary_configuration_guide.md)
- 壁面函数: [边界系统 v2](boundary_system_v2_upgrade.md)
- 移动地面: [配置指南](CONFIGURATION_GUIDE.md#边界条件配置)

#### 后处理
- 气动系数: [API 参考](API.md#calculate_coefficients)
- VTK 导出: [VTK 导出指南](VTK_EXPORT_GUIDE.md)
- 收敛分析: [算例教程](TUTORIALS.md#收敛性分析)

#### 性能优化
- CPU 优化: [性能优化](PERFORMANCE_OPTIMIZATION.md#cpu-性能优化)
- GPU 优化: [性能优化](PERFORMANCE_OPTIMIZATION.md#gpu-性能优化)
- 内存优化: [性能优化](PERFORMANCE_OPTIMIZATION.md#内存优化)

---

## 📊 文档统计

| 类别 | 文档数量 | 总页数 |
|------|---------|--------|
| 用户文档 | 8 | ~120 页 |
| 开发者文档 | 4 | ~80 页 |
| 技术文档 | 15+ | ~200 页 |
| 项目文档 | 20+ | ~300 页 |
| **总计** | **47+** | **~700 页** |

---

## 💡 文档改进建议

如果您发现以下问题，欢迎贡献改进：

- ❌ 文档缺失或过时
- ❌ 示例代码无法运行
- ❌ 表述不清或有歧义
- ❌ 缺少图示或流程图
- ❌ 链接失效

**贡献方式**：
1. Fork 仓库
2. 修改文档
3. 提交 Pull Request

---

## 📬 获取帮助

- **GitHub Issues**: [报告文档问题](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **GitHub Discussions**: [提问与讨论](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **项目联系人**: Mr Lu
- **邮箱联系**: luxw_chd@126.com

---

<div align="center">

**祝您学习愉快！** 🚀

*AutoFlowCFD 文档团队*

</div>

---

**最后更新**: 2026-07-25  
**版本**: AutoFlowCFD v0.1.0
