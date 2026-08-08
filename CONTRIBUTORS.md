# 贡献者名单

感谢所有为 AutoFlowCFD 做出贡献的个人和组织！

---

## 🌟 核心贡献者

| 姓名/ID | 角色 | 贡献领域 | GitHub |
|---------|------|---------|--------|
| AutoFlowCFD Team | 维护者 | 整体架构、核心开发 | [@AutoFlowCFD](https://github.com/AutoFlowCFD) |

---

## 💻 代码贡献者

### 核心模块开发

- **网格解析模块**: 
  - NAS 文件解析器实现
  - SoA 内存布局设计
  - 网格质量校验器

- **求解器引擎**:
  - FR 离散格式实现
  - CPU/GPU 后端开发
  - 湍流模型集成

- **边界条件系统**:
  - 边界条件框架设计
  - 壁面函数实现
  - 边界通量计算

- **后处理工具**:
  - 气动系数计算
  - VTK 数据导出
  - 收敛分析工具

### 基础设施

- **CI/CD 流水线**: GitHub Actions 配置
- **测试框架**: pytest 单元测试与集成测试
- **代码质量**: Black/Isort/MyPy 配置
- **依赖管理**: Poetry 项目配置

---

## 📖 文档贡献者

感谢以下文档的编写与维护：

- **用户文档**:
  - README.md
  - QUICKSTART.md
  - TUTORIALS.md
  - CONFIGURATION_GUIDE.md

- **开发者文档**:
  - ARCHITECTURE.md
  - DEVELOPER_GUIDE.md
  - API.md

- **技术文档**:
  - PERFORMANCE_OPTIMIZATION.md
  - 算例教程
  - 故障排查指南

---

## 🐛 问题报告者

感谢通过 GitHub Issues 报告问题和提出改进建议的用户：

- [查看已关闭的 Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues?q=is%3Aissue+is%3Aclosed)
- [查看当前开放的 Issues](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)

---

## 💡 功能建议者

感谢提出有价值功能建议的社区成员：

- CLI 命令行接口设计
- Python API 封装方案
- 配置管理系统架构
- AI Agent 集成思路

---

## 🌍 社区推广者

感谢帮助推广 AutoFlowCFD 的个人和组织：

- 技术博客作者
- 社交媒体分享者
- 会议演讲者
- 开源社区布道师

---

## 🤝 如何成为贡献者

我们欢迎各种形式的贡献！

### 代码贡献

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

### 文档贡献

- 修正拼写错误和语法问题
- 补充缺失的文档内容
- 添加示例代码和截图
- 翻译文档为其他语言

### 问题反馈

- 报告 Bug（附带复现步骤）
- 提出新功能建议
- 分享使用心得和最佳实践

### 社区帮助

- 回答其他用户的问题
- 分享算例和使用经验
- 参与技术讨论

---

## 📊 贡献统计

### GitHub 统计

- ⭐ Stars: [查看](https://github.com/AutoFlowCFD/AutoFlowCFD/stargazers)
- 🍴 Forks: [查看](https://github.com/AutoFlowCFD/AutoFlowCFD/network/members)
- 👥 Contributors: [查看](https://github.com/AutoFlowCFD/AutoFlowCFD/graphs/contributors)
- 💬 Discussions: [查看](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)

### 代码统计

```bash
# 查看代码行数
poetry run cloc src/

# 查看测试覆盖率
poetry run pytest --cov=autoflowcfd
```

---

## 🏆 贡献者权益

### 代码贡献者

- ✅ 名字列入 CONTRIBUTORS.md
- ✅ 获得 "Contributor" GitHub 标签
- ✅ 受邀加入组织（持续贡献者）

### 核心贡献者

- ✅ Commit 权限（经审核）
- ✅ 参与技术决策讨论
- ✅ 代表项目参加技术会议

---

## 🙏 致谢

特别感谢以下开源项目的卓越贡献，AutoFlowCFD 站在巨人的肩膀上：

- **NumPy/SciPy**: Python 科学计算基石
- **Numba**: CPU 并行加速利器
- **CuPy**: GPU 数值计算库
- **Click**: 优雅的 CLI 框架
- **VTK/PyVista**: 可视化数据处理
- **HDF5/h5py**: 高性能数据序列化
- **Loguru**: 现代化日志记录
- **pytest**: Python 测试框架

---

## 📬 联系我们

- **GitHub Issues**: [报告问题](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **GitHub Discussions**: [参与讨论](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)
- **项目联系人**: Mr Lu
- **邮箱**: luxw_chd@126.com

---

<div align="center">

**感谢您的贡献！** 🚀

*每一份贡献都让 AutoFlowCFD 变得更好*

</div>

---

**最后更新**: 2026-07-25  
**版本**: AutoFlowCFD v0.1.0
