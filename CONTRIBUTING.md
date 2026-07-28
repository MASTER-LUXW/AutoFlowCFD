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
1. 运行命令 `autoflowcfd solve --grid test.nas`
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
6. **提交代码**：遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范
   ```bash
   git commit -m "feat: add new turbulence model plugin"
   ```
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
# 运行单元测试
poetry run pytest tests/unit -v

# 运行代码检查
poetry run black --check src/
poetry run flake8 src/
poetry run mypy src/
```

---

## 代码规范

### Python代码风格

- 遵循 **PEP 8** 规范
- 使用 **Black** 自动格式化（行宽88字符）
- 使用 **isort** 排序import语句
- 强制类型注解，通过 **mypy** 严格检查

### 命名规范

- **模块/函数/变量**：`snake_case`（小写+下划线）
- **类名**：`PascalCase`（大驼峰）
- **常量**：`UPPER_CASE`（大写+下划线）
- **私有成员**：前缀 `_`（单下划线）

**示例**：
```python
# 模块名
autoflowcfd/grid/parser.py

# 类名
class GridParser:
    pass

# 函数名
def parse_nas_file(file_path: str) -> GridData:
    pass

# 变量名
node_count = 1000

# 常量
MAX_ITERATIONS = 10000

# 私有方法
def _validate_grid(self) -> bool:
    pass
```

### 注释规范

- **模块级docstring**：说明模块用途
- **函数docstring**：使用Google风格，包含参数、返回值、异常
- **行内注释**：解释复杂逻辑，避免冗余注释

**示例**：
```python
"""网格解析器模块

支持ANSA .nas文件格式解析（v22/v23/v24）。
"""

def parse_nodes(file_path: str) -> NodeArray:
    """解析节点数据
    
    Args:
        file_path: NAS文件路径
        
    Returns:
        NodeArray: 节点数组对象
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式错误
    """
    pass
```

### Git提交规范

遵循 **Conventional Commits** 规范：

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Type类型**：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构（非新功能、非Bug修复）
- `test`: 测试相关
- `chore`: 构建过程或辅助工具变动

**示例**：
```bash
git commit -m "feat(grid): add NAS v24 format support"
git commit -m "fix(solver): fix divergence issue in DES model"
git commit -m "docs(api): update Solver class docstring"
```

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
2. **更新版本号**：在 `pyproject.toml` 和 `__init__.py` 中更新
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
