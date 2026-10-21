# 贡献指南

感谢你对 **AutoFlowCFD** 项目的关注！我们欢迎所有形式的贡献，包括代码提交、文档改进、问题报告和功能建议。

## 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
  - [报告Bug](#报告bug)
  - [提出功能建议](#提出功能建议)
  - [提交代码](#提交代码)
- [开发环境搭建](#开发环境搭建)
- [代码规范](#代码规范)
- [提交流程](#提交流程)
- [Code Review标准](#code-review标准)
- [版本发布流程](#版本发布流程)

---

## 行为准则

本项目采用 [Contributor Covenant 行为准则](CODE_OF_CONDUCT.md)。参与本项目即表示你同意遵守该准则。请尊重所有贡献者，营造友好、包容的社区环境。

---

## 如何贡献

### 报告Bug

如果你发现了Bug，请通过 [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues) 提交Bug报告。

**Bug报告应包含**：
1. **标题**：简洁明了地描述问题
2. **环境信息**：
   - Python版本
   - 操作系统
   - AutoFlowCFD版本
   - 硬件配置（CPU/GPU型号）
3. **复现步骤**：详细列出复现问题的步骤
4. **预期行为**：说明期望的正确行为
5. **实际行为**：说明实际观察到的错误行为
6. **日志输出**：附上完整的错误日志和堆栈跟踪
7. **最小复现代码**：提供能够复现问题的最简代码示例

**Bug报告模板**：
```markdown
### 环境信息
- Python: 3.10.12
- OS: Ubuntu 22.04 LTS
- AutoFlowCFD: v0.1.0
- GPU: NVIDIA A100 40GB

### 复现步骤
1. 运行命令 `autoflowcfd solve steady volume.pkl --order 1 -o out`
2. 观察到...

### 预期行为
应该成功解析网格文件

### 实际行为
抛出 FileNotFoundError 异常

### 日志输出
```
Traceback (most recent call last):
  ...
```

### 附加信息
测试用的 .nas 文件已上传到附件
```

### 提出功能建议

我们欢迎新功能建议！请通过 [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues) 提交功能请求。

**功能建议应包含**：
1. **标题**：简洁描述功能
2. **背景**：说明为什么需要这个功能
3. **功能描述**：详细描述功能的具体内容
4. **使用场景**：举例说明如何使用该功能
5. **实现思路**（可选）：如果你有技术实现的想法，可以分享

### 提交代码

我们鼓励社区贡献代码！请遵循以下流程：

1. **Fork 仓库**：在GitHub上Fork本项目
2. **创建分支**：从 `dev` 分支创建功能分支
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```
3. **编写代码**：实现你的功能或修复
4. **编写测试**：为新代码添加单元测试
5. **更新文档**：如果修改了API，同步更新文档
6. **提交代码**：提交信息格式见下文 [提交信息](#提交信息)
7. **推送分支**：
   ```bash
   git push origin feature/your-feature-name
   ```
8. **创建Pull Request**：向 `dev` 分支提交PR

---

## 开发环境搭建

### 前置要求

- Python 3.10+
- Poetry（依赖管理工具）
- Git

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/AutoFlowCFD/AutoFlowCFD.git
cd AutoFlowCFD

# 安装Poetry
pip install poetry

# 安装项目依赖（含开发依赖）
poetry install

# 激活虚拟环境
poetry shell

# 安装pre-commit钩子
pre-commit install
```

### 验证安装

```bash
# 运行单元测试（-o addopts= 去掉缺省的覆盖率统计）
poetry run pytest tests/unit -q -o addopts=
poetry run autoflowcfd utils doctor
```

---

## 代码规范

项目约定以 [开发者指南](docs/DEVELOPER_GUIDE.md#约定评审按这些检查) 为准，要点：

- **同一语义只有一个事实来源**：常量、默认值、判据、枚举只定义一处，其余地方读它；发现重复就合并。
- **不静默兜底**：缺失的输入与元数据报错，不用缺省值继续；决定物理算例的参数写进 checkpoint、续算缺失即报错。
- **四个后端一致**：改 CPU / 单 GPU / CPU-MPI / 多 GPU 之一的算法，其余同步并加跨后端对照测试。
- **单文件不超过 500 行**，超过按功能拆分成子包。
- **注释与文档字符串用中文**（术语、公式、代码标识符除外），Google 风格的 Args / Returns / Raises；说明"为什么"。
- 命名：模块 / 函数 / 变量 `snake_case`，类 `PascalCase`，常量 `UPPER_CASE`，内部成员前缀 `_`。
- 行宽约 120 列；公共函数写类型注解。

### 测试

- 每个缺陷修复带一个在修改前失败、修改后通过的测试（行为测试优先于源码文本检查）。
- 提交前跑相关单元测试；改动 numba 核时用空的 `AFCFD_NUMBA_CACHE_ROOT` 再跑一遍。
- 改动命令行选项、配置字段、输出文件或 checkpoint 格式时同步更新 `docs/` 与 `--help`。

### 提交信息

首行 `V2 - <中文概述>（问题编号）`；正文写缺陷现象与根因、改动内容、测试结果（含修改前失败的核对）。一个提交只做一件事。

---

## 提交流程

### Pull Request流程

1. **确保代码通过CI检查**：
   - 单元测试覆盖率 ≥80%
   - Black格式化通过
   - flake8无警告
   - mypy类型检查通过

2. **填写PR模板**：
   ```markdown
   ## 变更类型
   - [ ] Bug修复
   - [ ] 新功能
   - [ ] 文档更新
   - [ ] 重构
   
   ## 变更描述
   简要描述本次PR的主要变更
   
   ## 相关Issue
   Closes #123
   
   ## 测试覆盖
   - [ ] 已添加单元测试
   - [ ] 已运行集成测试
   - [ ] 手动测试通过
   
   ## 截图（如适用）
   附上UI变更或性能对比截图
   ```

3. **等待Code Review**：
   - 至少需要1名Reviewer批准
   - 根据Review意见修改代码
   - 保持PR聚焦单一功能点

4. **合并到dev分支**：
   - Review通过后，由维护者合并
   - 删除功能分支

### Code Review标准

Reviewer将检查以下内容：

- [ ] 代码符合编码规范
- [ ] 功能实现符合需求
- [ ] 单元测试覆盖率达标
- [ ] 文档已同步更新
- [ ] 无明显的性能问题
- [ ] 异常处理完善
- [ ] 向后兼容性考虑

---

## 版本发布流程

### 版本号规范

遵循 [语义化版本](https://semver.org/lang/zh-CN/) (SemVer)：

- **主版本号**：不兼容的API修改
- **次版本号**：向下兼容的功能性新增
- **修订号**：向下兼容的问题修正

**示例**：`v0.1.0` → `v0.2.0` → `v1.0.0`

### 发布步骤

1. **更新CHANGELOG.md**：记录所有变更
2. **更新版本号**：只改 `pyproject.toml`（包内 `__version__` 从包元数据读取）
3. **打Tag**：
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0 - MVP version"
   git push origin v0.1.0
   ```
4. **创建Release**：在GitHub上创建Release，附上变更说明
5. **发布PyPI包**（可选）：
   ```bash
   poetry build
   poetry publish
   ```

---

## 常见问题

### Q: 我的PR为什么被拒绝？

A: 常见原因包括：
- 代码不符合编码规范
- 缺少单元测试
- 文档未更新
- PR范围过大，建议拆分为多个小PR
- 与项目路线图不符

### Q: 如何开始第一个贡献？

A: 推荐从以下任务入手：
1. 修复文档中的拼写错误
2. 改进现有代码的注释
3. 添加缺失的单元测试
4. 解决标记为 `good first issue` 的Issue

### Q: 我可以贡献CUDA代码吗？

A: 当然可以！但请注意：
- CUDA代码需遵循RAII内存管理原则
- 提供CPU后备方案
- 添加GPU特定单元测试
- 在文档中标明GPU依赖

---

## 联系方式

- **Issues**: [GitHub Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **Email**: contact@autoflowcfd.org

---

**再次感谢你的贡献！** 🎉
