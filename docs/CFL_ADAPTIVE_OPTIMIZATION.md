# CFL自适应策略优化指南

## 📊 问题诊断

### **原始问题**

从日志中观察到CFL调整过于频繁：

```
Iter 511: trend=-0.34 → CFL: 0.010 → 0.012  ✓ 增加
Iter 512: trend=+0.46 → CFL: 0.012 → 0.010  ✗ 立即减少
Iter 513: trend=-0.19 → CFL: 0.010 → 0.012  ✓ 又增加
Iter 514: trend=+0.36 → CFL: 0.012 → 0.010  ✗ 又减少
```

**影响：**
- ❌ CFL在0.010和0.012之间来回振荡
- ❌ 求解器无法稳定收敛
- ❌ Cd值持续上升（2.738 → 2.750）而非收敛

---

## 🔍 **根本原因分析**

### **原始CFL调整逻辑的问题**

```python
# 旧实现（有问题）
recent_trend = (res_history[-1] - res_history[-3]) / (res_history[-3] + 1e-30)

if recent_trend > 0.2:  # 阈值过低
    CFL *= 0.5
elif recent_trend < -0.15:  # 阈值过低
    CFL *= 1.2
```

**问题点：**

1. **窗口太小**：只比较3步残差，数值噪声太大
   - CFD残差本身有波动，3步不足以判断趋势
   
2. **阈值过低**：`trend > 0.2` 太敏感
   - 正常的数值波动就会触发调整
   - 导致频繁来回切换

3. **缺少滞后机制**：没有防止振荡的设计
   - 增加后立即减少，形成死循环

4. **线性趋势不适合残差**：残差通常是指数变化
   - 应该使用对数尺度更合理

---

## ✅ **优化方案**

### **改进后的CFL调整策略**

#### **1. 增大判断窗口**
```python
# 新实现：使用8步窗口
n_window = min(8, len(res_history))
recent = res_history[-n_window:]
```

**优势：**
- ✅ 平滑短期波动
- ✅ 更可靠的趋势判断
- ✅ 避免对单步异常的过度反应

---

#### **2. 使用对数趋势**
```python
# 计算对数尺度趋势
if recent[0] > 1e-30 and recent[-1] > 1e-30:
    log_trend = np.log(recent[-1] / recent[0]) / (n_window - 1)
```

**物理意义：**
- `log_trend = -0.1` → 每步下降约10%
- `log_trend = +0.1` → 每步上升约10%
- `log_trend = 0.0` → 基本不变

**优势：**
- ✅ 适合残差的指数衰减/增长特性
- ✅ 对不同量级的残差有一致的灵敏度
- ✅ 更容易设置合理的阈值

---

#### **3. 提高调整阈值**
```python
# 更保守的阈值
if log_trend > 0.15:  # 原0.2 → 现0.15（但基于8步窗口，实际更严格）
    CFL *= 0.6  # 原×0.5 → 现×0.6（更温和）
    
elif log_trend < -0.25:  # 原-0.15 → 现-0.25（更严格）
    if decrease_ratio > 0.7:  # 新增：要求70%的步都在下降
        CFL *= 1.15  # 原×1.2 → 现×1.15（更温和）
```

**对比：**

| 参数 | 旧值 | 新值 | 效果 |
|------|------|------|------|
| 窗口大小 | 3步 | 8步 | 更平滑 |
| 增加阈值 | trend > 0.2 | log_trend > 0.15 | 更稳定 |
| 减少阈值 | trend < -0.15 | log_trend < -0.25 | 更保守 |
| 增加幅度 | ×1.2 | ×1.15 | 更温和 |
| 减少幅度 | ×0.5 | ×0.6 | 更温和 |

---

#### **4. 添加滞后机制**
```python
# 检查持续性：至少70%的步在下降
decreases = sum(1 for i in range(len(recent)-1) 
               if recent[i+1] < recent[i])
decrease_ratio = decreases / (len(recent) - 1)

if decrease_ratio > 0.7:  # 只有持续下降才增加CFL
    CFL *= 1.15
```

**优势：**
- ✅ 避免短暂下降就增加CFL
- ✅ 确保趋势是持续的
- ✅ 防止振荡行为

---

#### **5. 延迟启动**
```python
# 前10步不调整CFL，让求解器稳定
if iteration > 10 and len(res_history) >= 8:
    # 执行CFL调整
```

**原因：**
- 初始阶段残差波动大
- 需要时间建立稳定的趋势
- 避免过早干预

---

## 📈 **预期效果对比**

### **旧策略（你的日志）**
```
Iter 511: CFL 0.010 → 0.012  (trend=-0.34)
Iter 512: CFL 0.012 → 0.010  (trend=+0.46)  ← 立即反转
Iter 513: CFL 0.010 → 0.012  (trend=-0.19)  ← 又反转
Iter 514: CFL 0.012 → 0.010  (trend=+0.36)  ← 持续振荡
...
结果：Cd持续上升，无法收敛
```

### **新策略（预期）**
```
Iter 511-518: CFL保持0.010  (观察8步趋势)
Iter 519: 如果8步平均log_trend < -0.25且70%下降
          → CFL 0.010 → 0.012
Iter 520-527: CFL保持0.012  (继续观察)
Iter 528: 如果趋势变差(log_trend > 0.15)
          → CFL 0.012 → 0.007
...
结果：CFL稳定，残差平稳下降，Cd收敛
```

---

## 🎯 **关键改进总结**

| 改进项 | 说明 | 收益 |
|--------|------|------|
| **窗口增大** | 3步→8步 | 减少噪声干扰 |
| **对数趋势** | 线性→对数 | 更适合残差特性 |
| **阈值提高** | 0.2→0.15/-0.25 | 减少误触发 |
| **滞后机制** | 70%持续性检查 | 防止振荡 |
| **调整温和** | ×0.5/1.2→×0.6/1.15 | 更平滑过渡 |
| **延迟启动** | 前10步不调整 | 避免初期干扰 |

---

## 🔧 **如何验证优化效果**

### **运行仿真并观察日志**

```bash
autoflowcfd solve steady \
  --grid plate.nas \
  --backend gpu \
  --order 3 \
  --max-iter 2500 \
  --output results/
```

### **期望看到的日志**

#### **✅ 正常情况（CFL稳定）**
```
Iter  511/2500 | Res(rel): 9.3126e+00 | Cd: 2.7384 | Cl: -0.1679
Iter  512/2500 | Res(rel): 2.4678e+01 | Cd: 2.7411 | Cl: -0.1679
Iter  513/2500 | Res(rel): 9.6441e+00 | Cd: 2.7442 | Cl: -0.1680
...
[CFL STATUS] log_trend=-0.05/step, CFL=0.010, no adjustment needed
...
[CFL ADJUST] Residuals decreasing well (log_trend=-0.28/step, 
             decrease_ratio=87%), increasing CFL: 0.010 -> 0.012
```

**特征：**
- ✅ CFL调整频率大幅降低（每50-100步一次）
- ✅ 调整后能保持稳定一段时间
- ✅ 残差整体呈下降趋势
- ✅ Cd值趋于稳定

#### **❌ 异常情况（仍然振荡）**
```
[CFL ADJUST] reducing CFL: 0.010 -> 0.006
[CFL ADJUST] increasing CFL: 0.006 -> 0.007
[CFL ADJUST] reducing CFL: 0.007 -> 0.004
```

**可能原因：**
1. 网格质量差（高纵横比单元）
2. 边界条件不合理
3. 湍流模型参数不当
4. 初始场太差

**解决：**
- 手动设置固定CFL：`--cfl-init 0.05 --cfl-max 0.1`
- 检查网格质量
- 改善初始条件

---

## 💡 **进一步优化建议**

### **1. 根据残差量级动态调整阈值**

```python
# 残差很大时（早期），允许更大波动
if res_history[-1] > 1.0:
    increase_threshold = -0.3  # 更保守
    decrease_threshold = 0.2   # 更宽松
else:
    increase_threshold = -0.25  # 标准
    decrease_threshold = 0.15   # 标准
```

### **2. 分阶段策略**

```python
# Phase 1: 初始阶段（残差>1.0）- 保守
if res_history[-1] > 1.0:
    max_cfl_increase = 1.1
    min_cfl_decrease = 0.7

# Phase 2: 中期阶段（0.01<残差<1.0）- 标准
elif res_history[-1] > 0.01:
    max_cfl_increase = 1.15
    min_cfl_decrease = 0.6

# Phase 3: 后期阶段（残差<0.01）- 激进
else:
    max_cfl_increase = 1.2
    min_cfl_decrease = 0.5
```

### **3. 监测Cd稳定性**

```python
# 如果Cd已经稳定，停止CFL调整
if len(cd_history) >= 20:
    cd_recent = cd_history[-20:]
    cd_std = np.std(cd_recent) / np.mean(cd_recent)
    
    if cd_std < 0.01:  # Cd波动<1%，认为已稳定
        logger.info("Cd stabilized, freezing CFL at current value")
        disable_cfl_adjustment = True
```

---

## 📚 **相关代码位置**

- **CFL调整实现**: [`src/autoflowcfd/core/solver_steady.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\solver_steady.py) (第476-530行)
- **时间积分器**: [`src/autoflowcfd/core/time_integration.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\time_integration.py)
- **收敛监控**: [`src/autoflowcfd/core/convergence.py`](d:\myWorkspace\AutoFlowCFD\src\autoflowcfd\core\convergence.py)

---

## ✅ **总结**

**核心改进：**
1. ✅ 窗口从3步增加到8步
2. ✅ 使用对数趋势代替线性趋势
3. ✅ 提高阈值减少误触发
4. ✅ 添加70%持续性检查
5. ✅ 调整幅度更温和（×0.6/1.15）
6. ✅ 前10步不调整

**预期效果：**
- CFL调整频率降低80%以上
- 残差收敛更平稳
- Cd值更快达到稳定
- 总迭代次数减少10-20%

**下一步：**
重新运行仿真，观察新的CFL调整行为。如果仍有问题，考虑手动设置固定CFL或进一步调整阈值。

---

**最后更新**: 2026-07-31  
**维护者**: AutoFlowCFD Team
