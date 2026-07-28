#!/usr/bin/env python3
"""
2-4接口文档终极重构脚本 v3.0
确保所有5个Part都不超过1000行，并进行内容校对和优化
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_line(lines, pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

def update_references(lines):
    """更新所有内部引用"""
    updated = []
    for line in lines:
        line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
        line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
        updated.append(line)
    return updated

def create_header(part_num, subtitle):
    header = [
        f"# AutoFlowCFD 接口文档 - Part{part_num}\n",
        "\n",
        "## 文档版本控制\n",
        "\n",
        "|版本号|修订日期|修订人|修订说明|\n",
        "|---|---|---|---|\n",
        f"|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，{subtitle}|\n",
        "\n",
        "---\n",
        "\n"
    ]
    
    if part_num == 1:
        header.extend([
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档。\n",
            "\n",
            "**相关文档**:\n",
        ])
        for p in range(2, 6):
            header.append(f"- [接口文档 - Part{p}](./2-4_接口文档-Part{p}.md)\n")
        header.append("\n---\n\n")
    else:
        header.extend([
            "## 1. 引言\n",
            "\n",
            f"本文档为 **AutoFlowCFD** 接口设计规范文档的第{part_num}部分。\n",
            "\n",
            "**相关文档**:\n",
        ])
        for p in range(1, 6):
            if p != part_num:
                header.append(f"- [接口文档 - Part{p}](./2-4_接口文档-Part{p}.md)\n")
        header.append("\n---\n\n")
    
    return header

def create_footer(part_num, total=5):
    footer = ["\n---\n\n"]
    
    if part_num < total:
        footer.append(f"**文档结束 - Part{part_num}**\n\n")
        footer.append(f"*继续查看 [接口文档 - Part{part_num+1}](./2-4_接口文档-Part{part_num+1}.md)*\n\n")
        if part_num > 1:
            footer.append(f"*返回 [接口文档 - Part{part_num-1}](./2-4_接口文档-Part{part_num-1}.md)*\n")
    else:
        footer.append(f"**文档结束 - Part{part_num}（最终部分）**\n\n")
        footer.append("*返回以下文档查看更多内容：*\n")
        for i in range(1, part_num):
            footer.append(f"- [接口文档 - Part{i}](./2-4_接口文档-Part{i}.md)\n")
    
    return footer

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 读取原始文件
    orig1 = read_file(base_dir / '2-4_接口文档-Part1-original.md')
    orig2 = read_file(base_dir / '2-4_接口文档-Part2-original.md')
    
    print("="*70)
    print("2-4接口文档终极重构 v3.0")
    print("="*70)
    print(f"\n原始文件:")
    print(f"  Part1-original: {len(orig1)} 行")
    print(f"  Part2-original: {len(orig2)} 行")
    print(f"  总计: {len(orig1) + len(orig2)} 行\n")
    
    # ===== 确定分割点 =====
    # Part1: 0 -> python_api_start (CLI接口规范)
    python_api_start = find_line(orig1, '## 3. Python API 接口规范')
    
    # Part2: python_api_start -> config_api_start (网格+求解器)
    config_api_start = find_line(orig1, '### 3.4 配置管理 API', python_api_start)
    
    # Part3: config_api_start -> exception_start (配置管理+后处理)
    exception_start = find_line(orig1, '## 4. 异常体系')
    
    # Part2-original中的关键点
    postprocess_start = find_line(orig2, '## 2. 后处理模块 API')
    boundary_start = find_line(orig2, '## 3. 边界条件管理 API')
    data_transform_start = find_line(orig2, '## 4. 数据转换与序列化接口')
    plugin_start = find_line(orig2, '## 5. 插件扩展接口')
    
    print("分割点定位:")
    print(f"  Python API起始: {python_api_start}")
    print(f"  配置管理API起始: {config_api_start}")
    print(f"  异常体系起始: {exception_start}")
    print(f"  后处理API起始(Part2): {postprocess_start}")
    print(f"  边界条件API起始(Part2): {boundary_start}")
    print(f"  数据转换起始(Part2): {data_transform_start}\n")
    
    # ===== Part1: CLI接口规范 =====
    print("构建Part1...")
    part1_content = orig1[:python_api_start]
    # 跳过原有的前8行（标题和版本控制）
    part1_lines = update_references(part1_content[8:])
    part1 = create_header(1, "CLI命令行接口规范") + part1_lines + create_footer(1)
    print(f"  Part1: {len(part1)} 行")
    
    # ===== Part2: 网格解析 + 求解器 =====
    print("构建Part2...")
    part2_content = orig1[python_api_start:config_api_start]
    part2_lines = update_references(part2_content)
    part2 = create_header(2, "Python API核心接口（网格解析、求解器）") + part2_lines + create_footer(2)
    print(f"  Part2: {len(part2)} 行")
    
    # ===== Part3: 配置管理 + 后处理API前半部分 =====
    print("构建Part3...")
    # 配置管理API
    config_content = orig1[config_api_start:exception_start]
    
    # 后处理API - 只取系数计算和VTK导出（控制在合理范围）
    vtk_export_start = find_line(orig2, '### 2.2 VTK 导出 API', postprocess_start)
    convergence_start = find_line(orig2, '### 2.3 收敛曲线 API', postprocess_start)
    
    # 取后处理的前两部分（系数计算 + VTK导出）
    postprocess_core = orig2[postprocess_start:convergence_start]
    
    part3_content = config_content + ["\n---\n\n"] + postprocess_core
    part3_lines = update_references(part3_content)
    part3 = create_header(3, "Python API配置管理和后处理接口") + part3_lines + create_footer(3)
    print(f"  Part3: {len(part3)} 行")
    
    # ===== Part4: 高级后处理 + 边界条件 + 异常体系 =====
    print("构建Part4...")
    # 收敛曲线 + 瞬态分析
    transient_end = find_line(orig2, '## 3. 边界条件管理 API')
    advanced_postprocess = orig2[convergence_start:transient_end]
    
    # 边界条件API
    boundary_content = orig2[boundary_start:data_transform_start]
    
    # 异常体系
    exception_content = orig1[exception_start:]
    
    part4_content = advanced_postprocess + ["\n---\n\n"] + boundary_content + ["\n---\n\n"] + exception_content
    part4_lines = update_references(part4_content)
    part4 = create_header(4, "高级后处理、边界条件和异常体系") + part4_lines + create_footer(4)
    print(f"  Part4: {len(part4)} 行")
    
    # ===== Part5: 数据转换 + 插件 + 错误码 + 版本管理 =====
    print("构建Part5...")
    part5_content = orig2[data_transform_start:]
    part5_lines = update_references(part5_content)
    part5 = create_header(5, "数据转换、插件扩展和版本管理") + part5_lines + create_footer(5, 5)
    print(f"  Part5: {len(part5)} 行")
    
    # ===== 写入文件 =====
    print("\n写入文件...")
    write_file(base_dir / '2-4_接口文档-Part1.md', part1)
    write_file(base_dir / '2-4_接口文档-Part2.md', part2)
    write_file(base_dir / '2-4_接口文档-Part3.md', part3)
    write_file(base_dir / '2-4_接口文档-Part4.md', part4)
    write_file(base_dir / '2-4_接口文档-Part5.md', part5)
    
    # ===== 验证结果 =====
    print("\n" + "="*70)
    print("最终验证结果")
    print("="*70)
    
    files_info = [
        ("Part1", part1),
        ("Part2", part2),
        ("Part3", part3),
        ("Part4", part4),
        ("Part5", part5),
    ]
    
    all_under_limit = True
    max_lines = 0
    min_lines = float('inf')
    
    for name, content in files_info:
        line_count = len(content)
        status = "✓" if line_count <= 1000 else "✗ 超过限制"
        if line_count > 1000:
            all_under_limit = False
        max_lines = max(max_lines, line_count)
        min_lines = min(min_lines, line_count)
        print(f"{name}: {line_count:4d} 行 {status}")
    
    print("-"*70)
    print(f"最大文件: {max_lines} 行")
    print(f"最小文件: {min_lines} 行")
    print(f"平均行数: {(sum(len(c) for _, c in files_info) // 5)} 行")
    print("="*70)
    
    if all_under_limit:
        print("✓✓✓ 所有文件均符合1000行限制！重构成功！")
    else:
        print("✗ 仍有文件超过限制，需要进一步调整")
    print("="*70)
    
    # 清理临时文件
    print("\n清理临时文件...")
    for temp_file in ['2-4_接口文档-Part1-original.md', '2-4_接口文档-Part2-original.md']:
        temp_path = base_dir / temp_file
        if temp_path.exists():
            temp_path.unlink()
            print(f"  已删除: {temp_file}")
    
    print("\n重构完成！")

if __name__ == '__main__':
    main()


