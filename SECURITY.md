# 安全策略

## 报告漏洞

AutoFlowCFD团队高度重视项目安全性。如果你发现安全漏洞，请**不要**通过公开Issue报告，而应通过以下方式私下联系我们：

- **邮箱**：security@autoflowcfd.org
- **GitHub Security Advisories**：[创建安全公告](https://github.com/AutoFlowCFD/AutoFlowCFD/security/advisories/new)

我们将在 **48小时内** 确认收到报告，并在 **7天内** 提供初步评估和修复计划。

## 支持版本

我们仅为最新版本提供安全更新：

| 版本 | 支持状态 |
|------|---------|
| v0.1.x (最新) | ✅ 积极支持 |
| v0.0.x | ❌ 不再支持 |

**建议**：始终使用最新稳定版本以获得安全补丁。

## 安全开发准则

### 1. 输入验证

所有外部输入必须经过严格验证：

```python
# ✅ 正确：验证文件路径
def parse_grid(file_path: str) -> GridData:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {file_path}")
    if not path.suffix == '.nas':
        raise ValueError("Only .nas files are supported")
    # ... 解析逻辑

# ❌ 错误：未验证输入
def parse_grid(file_path: str):
    with open(file_path, 'r') as f:  # 可能存在路径遍历攻击
        pass
```

### 2. 依赖安全管理

- 定期运行 `poetry update` 更新依赖
- 使用 `pip-audit` 扫描已知漏洞：
  ```bash
  pip install pip-audit
  pip-audit
  ```
- CI流水线自动执行依赖安全检查

### 3. 敏感信息管理

- **禁止**在代码中硬编码密钥、密码或令牌
- 使用环境变量或配置文件管理敏感信息
- 将 `.env` 文件添加到 `.gitignore`

```python
# ✅ 正确：从环境变量读取
import os
api_key = os.getenv("API_KEY")

# ❌ 错误：硬编码密钥
api_key = "sk-1234567890abcdef"
```

### 4. CUDA内存安全

- 遵循RAII原则管理GPU内存
- 确保异常情况下正确释放资源
- 避免内存泄漏和悬空指针

```cpp
// ✅ 正确：使用智能指针管理CUDA内存
class CUDABuffer {
private:
    float* d_data;
    size_t size;
    
public:
    CUDABuffer(size_t n) : size(n) {
        cudaMalloc(&d_data, n * sizeof(float));
    }
    
    ~CUDABuffer() {
        if (d_data) cudaFree(d_data);
    }
    
    // 禁止拷贝
    CUDABuffer(const CUDABuffer&) = delete;
    CUDABuffer& operator=(const CUDABuffer&) = delete;
};
```

### 5. 并行计算安全

- CPU多线程：使用线程安全的数据结构
- GPU计算：确保kernel launch参数合法
- 避免竞态条件和数据竞争

## 已知安全问题

我们会在以下位置公布已知安全问题及缓解措施：

- [GitHub Security Advisories](https://github.com/AutoFlowCFD/AutoFlowCFD/security/advisories)
- [SECURITY.md](SECURITY.md) 本文件

## 安全更新流程

1. **接收报告**：通过私有渠道接收漏洞报告
2. **验证问题**：复现并确认漏洞存在
3. **评估影响**：确定漏洞严重性和影响范围
4. **制定修复方案**：设计最小影响的修复方案
5. **实施修复**：在私有分支中修复
6. **测试验证**：全面测试修复效果
7. **发布公告**：发布安全公告和补丁版本
8. **致谢贡献者**：公开感谢报告者（经同意）

## 依赖项安全扫描

CI流水线自动执行以下安全检查：

```yaml
# .github/workflows/security-scan.yml
name: Security Scan
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: poetry install
        
      - name: Run pip-audit
        run: |
          pip install pip-audit
          pip-audit --format json --output audit-results.json
          
      - name: Check for vulnerabilities
        run: |
          if jq -e '.[] | select(.severity == "HIGH" or .severity == "CRITICAL")' audit-results.json; then
            echo "Found high/critical vulnerabilities!"
            exit 1
          fi
```

## 最佳实践

### 对于用户

1. **保持更新**：及时升级到最新版本
2. **验证输入**：对输入的网格文件进行质量检查
3. **隔离环境**：在虚拟环境或容器中运行
4. **备份数据**：定期备份重要仿真数据
5. **监控日志**：关注异常日志输出

### 对于开发者

1. **遵循规范**：严格遵守编码规范和安全准则
2. **代码审查**：所有PR必须经过安全审查
3. **最小权限**：仅请求必要的系统权限
4. **文档同步**：安全相关的代码变更必须更新文档
5. **持续学习**：关注CFD软件安全领域的最新进展

## 联系方式

- **安全邮箱**：security@autoflowcfd.org
- **GitHub Issues**：[非安全问题](https://github.com/AutoFlowCFD/AutoFlowCFD/issues)
- **社区讨论**：[GitHub Discussions](https://github.com/AutoFlowCFD/AutoFlowCFD/discussions)

---

**最后更新**：2026-07-23
