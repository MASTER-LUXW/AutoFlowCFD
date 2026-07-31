# 日志输出格式优化

## 📊 优化内容

### **改进前的输出格式**

```
[Iter 533] Cd breakdown: pressure=2.8015, friction=0.0005, total=2.8020

Iter   533/2500 | Res(rel): 1.6941e+01 | Cd: 2.8020 | Cl: -0.1692
```

**问题：**
- ❌ Cd breakdown信息在第一行，与主迭代信息分离
- ❌ 格式不统一，难以快速扫描
- ❌ 视觉层次不清晰

---

### **改进后的输出格式**

```
Iter   533/2500  |  Res(rel): 1.6941e+01  |  Cd: 2.8020  |  Cl: -0.1692
                  Cd breakdown: pressure=2.8015, friction=0.0005

Iter   534/2500  |  Res(rel): 1.7123e+01  |  Cd: 2.8045  |  Cl: -0.1693
                  Cd breakdown: pressure=2.8040, friction=0.0005
```

**优势：**
- ✅ 主信息在一行，简洁明了
- ✅ Cd breakdown与第一个 `|` 对齐，视觉整齐
- ✅ 使用双竖线 `|` 分隔主要字段，更易读
- ✅ 便于快速扫描关键数据

---

## 🔧 **代码修改**

### **1. 修改 [`aero_coeffs.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\aero_coeffs.py)**

**原返回值：**
```python
return float(Cd), float(Cl)
```

**新返回值：**
```python
return float(Cd), float(Cl), float(Cd_p), float(Cd_f)
```

**说明：**
- 添加 `Cd_p`（压力阻力系数）和 `Cd_f`（摩擦阻力系数）到返回值
- 移除内部的logger.info调用，由调用者统一格式化输出

---

### **2. 修改 [`solver_steady.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\solver_steady.py)**

**原代码：**
```python
Cd, Cl = self.aero_calculator.compute_coefficients(...)

logger.info(
    f"Iter {iteration:5d}/{actual_max_iter} | "
    f"Res(rel): {rel_res:.4e} | "
    f"Cd: {Cd:.4f} | "
    f"Cl: {Cl:.4f}"
)
```

**新代码：**
```python
Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(...)

# Main line: iteration info
logger.info(
    f"Iter {iteration:5d}/{actual_max_iter}  |  "
    f"Res(rel): {rel_res:.4e}  |  "
    f"Cd: {Cd:.4f}  |  "
    f"Cl: {Cl:.4f}"
)

# Second line: Cd breakdown (aligned with first `|`)
prefix_len = len(f"Iter {iteration:5d}/{actual_max_iter}")
logger.info(
    f"{'':>{prefix_len + 2}s}  "
    f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
)
```

**说明：**
- 接收4个返回值：`Cd, Cl, Cd_p, Cd_f`
- 第一行输出主要迭代信息（残差、Cd、Cl）
- 第二行缩进输出Cd分解信息
- 使用 `{'':28s}` 创建28字符的空白缩进

---

### **3. 修改 [`transient_solver_loop.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\transient_solver_loop.py)**

同步更新瞬态求解器的输出格式：

```
Cd, Cl, Cd_p, Cd_f = self.aero_calculator.compute_coefficients(...)

if self.n_steps % max(1, self.config.sample_interval) == 0 or self.n_steps == 1:
    logger.info(
        f"Step {self.n_steps:6d} | t={self.current_time:.6f}s | "
        f"Cd={Cd:.4f} | Cl={Cl:.4f}"
    )
    # 第二行：Cd分解（与第一个|对齐）
    prefix_len = len(f"Step {self.n_steps:6d}")
    logger.info(
        f"{'':>{prefix_len + 2}s}  "
        f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
    )
```

---

## 📈 **效果对比**

### **稳态求解器输出示例**

#### **改进前**
```
[Iter 511] Cd breakdown: pressure=2.7380, friction=0.0005, total=2.7384
Iter   511/2500 | Res(rel): 9.3126e+00 | Cd: 2.7384 | Cl: -0.1679

[Iter 512] Cd breakdown: pressure=2.7406, friction=0.0005, total=2.7411
Iter   512/2500 | Res(rel): 2.4678e+01 | Cd: 2.7411 | Cl: -0.1679
```

#### **改进后**
```
Iter   511/2500  |  Res(rel): 9.3126e+00  |  Cd: 2.7384  |  Cl: -0.1679
                  Cd breakdown: pressure=2.7380, friction=0.0005

Iter   512/2500  |  Res(rel): 2.4678e+01  |  Cd: 2.7411  |  Cl: -0.1679
                  Cd breakdown: pressure=2.7406, friction=0.0005
```

---

### **瞬态求解器输出示例**

#### **改进前**
```
Step      1 | t=0.000100s | Cd=0.2850 | Cl=0.0120
[Iter 1] Cd breakdown: pressure=0.2845, friction=0.0005, total=0.2850
```

#### **改进后**
```
Step      1 | t=0.000100s | Cd=0.2850 | Cl=0.0120
          Cd breakdown: pressure=0.2845, friction=0.0005
```

---

## 🎯 **设计原则**

### **1. 信息层次化**
- **第一行**：核心指标（迭代步、残差、气动力系数）
- **第二行**：详细分解（阻力组成）

### **2. 视觉对齐**
- **缩进对齐** | Cd breakdown与第一个`|`垂直对齐 | 视觉整齐 |

### **3. 减少冗余**
- 移除 `[Iter XXX]` 前缀（已在第一行显示）
- 移除 `total=` 字段（与第一行的 `Cd:` 重复）

### **4. 一致性**
- 稳态和瞬态求解器采用相同的格式风格
- 所有日志输出遵循统一的缩进规范

---

## 💡 **进一步优化建议**

### **1. 可选的详细模式**

```python
# 启用详细模式时显示更多信息
if self.config.verbose_logging:
    logger.info(
        f"{'':28s}  "
        f"Cd breakdown: pressure={Cd_p:.4f} ({Cd_p/Cd*100:.1f}%), "
        f"friction={Cd_f:.4f} ({Cd_f/Cd*100:.1f}%)"
    )
```

**输出：**
```
Iter   533/2500  |  Res(rel): 1.6941e+01  |  Cd: 2.8020  |  Cl: -0.1692
                            Cd breakdown: pressure=2.8015 (99.98%), friction=0.0005 (0.02%)
```

### **2. 条件性输出**

```python
# 只在特定条件下输出breakdown
if iteration <= 10 or iteration % 10 == 0 or abs(Cd_f/Cd) > 0.1:
    logger.info(
        f"{'':28s}  "
        f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
    )
```

**优势：**
- 减少日志行数（每10步或摩擦占比>10%时输出）
- 突出异常情况（摩擦阻力异常大时）

### **3. 颜色编码（终端支持时）**

```python
from loguru import logger

# 根据摩擦占比设置颜色
friction_ratio = abs(Cd_f / Cd) if Cd != 0 else 0

if friction_ratio > 0.1:
    # 异常：黄色警告
    logger.warning(
        f"{'':28s}  "
        f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f} ⚠️"
    )
else:
    # 正常：普通信息
    logger.info(
        f"{'':28s}  "
        f"Cd breakdown: pressure={Cd_p:.4f}, friction={Cd_f:.4f}"
    )
```

---

## 📝 **相关文件**

- **气动系数计算**: [`src/autoflowcfd/core/aero_coeffs.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\aero_coeffs.py)
- **稳态求解器**: [`src/autoflowcfd/core/solver_steady.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\solver_steady.py)
- **瞬态求解器**: [`src/autoflowcfd/core/transient_solver_loop.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\transient_solver_loop.py)

---

## ✅ **总结**

**改进要点：**
1. ✅ 将Cd breakdown移至第二行缩进显示
2. ✅ 使用双竖线分隔主要字段
3. ✅ 移除冗余信息（total字段）
4. ✅ 统一稳态和瞬态求解器的输出格式
5. ✅ 提高日志可读性和扫描效率

**预期效果：**
- 日志更简洁，关键信息一目了然
- 视觉层次清晰，便于快速定位问题
- 格式统一，提升整体专业性

---

**最后更新**: 2026-07-31  
**维护者**: AutoFlowCFD Team
