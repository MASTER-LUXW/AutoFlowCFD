#!/usr/bin/env python3
"""
2-4接口文档精确重构脚本 v2.0
基于原始Part1(2417行)和Part2(1519行)，精确拆分为5个文件，每个不超过1000行

拆分方案:
- Part1: 引言 + CLI接口规范 (目标: ~850行)
- Part2: Python API - 网格解析 + 求解器 (目标: ~650行)  
- Part3: Python API - 配置管理 + 后处理API (目标: ~850行)
- Part4: Python API - 边界条件 + 异常体系 (目标: ~700行)
- Part5: 数据转换 + 插件扩展 + 错误码 + 版本管理 (目标: ~650行)
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_line_number(lines, pattern, start=0):
    """查找包含特定模式的行号"""
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

def create_doc_header(part_num, subtitle, related_parts):
    """创建文档头部"""
    header = [
        f"# AutoFlowCFD 接口文档 - Part{part_num}\n",
        "\n",
        "## 文档版本控制\n",
        "\n",
        "|版本号|修订日期|修订人|修订说明|\n",
        "|---|---|---|---|\n",
        "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，" + subtitle + "|\n",
        "\n",
        "---\n",
        "\n"
    ]
    
    if part_num == 1:
        header.extend([
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD**（汽车外流场专用开源 CFD 软件）的接口设计规范文档。\n",
            "\n",
            "**相关文档**:\n",
        ])
        for p in related_parts:
            if p != part_num:
                header.append(f"- [接口文档 - Part{p}](./2-4_接口文档-Part{p}.md)\n")
        header.append("\n---\n\n")
    
    return header

def create_doc_footer(part_num, total_parts=5):
    """创建文档尾部"""
    footer = ["\n---\n\n"]
    
    if part_num < total_parts:
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
    orig_part1 = read_file(base_dir / '2-4_接口文档-Part1-original.md')
    orig_part2 = read_file(base_dir / '2-4_接口文档-Part2-original.md')
    
    print(f"原始文件行数:")
    print(f"Part1-original: {len(orig_part1)}")
    print(f"Part2-original: {len(orig_part2)}")
    print(f"总计: {len(orig_part1) + len(orig_part2)} 行\n")
    
    # ===== Part1: 引言 + CLI接口规范 =====
    # 找到Python API开始位置
    python_api_start = find_line_number(orig_part1, '## 3. Python API 接口规范')
    print(f"Python API起始行(Part1): {python_api_start}")
    
    # Part1内容: 从开头到Python API之前
    part1_content = orig_part1[:python_api_start]
    
    # 更新Part1的版本信息
    part1_lines = []
    for line in part1_content:
        if '|v0.1|2026-07-22|' in line:
            line = '|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，CLI接口规范|\n'
        part1_lines.append(line)
    
    part1_header = create_doc_header(1, "CLI命令行接口规范", [2, 3, 4, 5])
    # 跳过原有的标题和版本控制（前8行）
    final_part1 = part1_header + part1_lines[8:] + create_doc_footer(1)
    
    print(f"Part1行数: {len(final_part1)}")
    
    # ===== Part2: Python API - 网格解析 + 求解器 =====
    # 找到配置管理API开始位置
    config_api_start = find_line_number(orig_part1, '### 3.4 配置管理 API', python_api_start)
    print(f"配置管理API起始行(Part1): {config_api_start}")
    
    # Part2内容: 从Python API开始到配置管理之前
    part2_content = orig_part1[python_api_start:config_api_start]
    
    part2_header = create_doc_header(2, "Python API核心接口（网格解析、求解器）", [1, 3, 4, 5])
    final_part2 = part2_header + part2_content + create_doc_footer(2)
    
    print(f"Part2行数: {len(final_part2)}")
    
    # ===== Part3: Python API - 配置管理 + 后处理API =====
    # 找到边界条件API开始位置（在Part2-original中）
    boundary_api_start = find_line_number(orig_part2, '## 3. 边界条件管理 API')
    print(f"边界条件API起始行(Part2-orig): {boundary_api_start}")
    
    # 找到后处理API开始位置
    postprocess_start = find_line_number(orig_part2, '## 2. 后处理模块 API')
    print(f"后处理API起始行(Part2-orig): {postprocess_start}")
    
    # Part3内容: 配置管理API + 后处理API（到VTK导出之前）
    config_to_end = orig_part1[config_api_start:]
    
    # 在后处理API中找到合适的分割点（VTK导出API之前）
    vtk_export_in_part2 = find_line_number(orig_part2, '### 2.2 VTK 导出 API', postprocess_start)
    print(f"VTK导出API起始行(Part2-orig): {vtk_export_in_part2}")
    
    if vtk_export_in_part2 > 0:
        postprocess_core = orig_part2[postprocess_start:vtk_export_in_part2]
        part3_content = config_to_end + ["\n---\n\n"] + postprocess_core
        
        part3_header = create_doc_header(3, "Python API配置管理和气动系数计算", [1, 2, 4, 5])
        final_part3 = part3_header + part3_content + create_doc_footer(3)
        
        print(f"Part3行数: {len(final_part3)}")
    else:
        print("警告: 未找到VTK导出API位置")
        final_part3 = []
    
    # ===== Part4: Python API - 高级后处理 + 边界条件 + 异常体系 =====
    # Part4内容: VTK导出 + 收敛曲线 + 瞬态分析 + 边界条件 + 异常体系
    
    # 从Part2-original提取VTK导出及之后的后处理API
    transient_analyzer_start = find_line_number(orig_part2, '#### 2.4.1 API-POST-004: TransientAnalyzer', postprocess_start)
    print(f"瞬态分析API起始行(Part2-orig): {transient_analyzer_start}")
    
    if vtk_export_in_part2 > 0 and transient_analyzer_start > 0:
        advanced_postprocess = orig_part2[vtk_export_in_part2:transient_analyzer_start + 50]  # 包含瞬态分析
        
        # 添加边界条件API
        data_transform_start = find_line_number(orig_part2, '## 4. 数据转换与序列化接口')
        print(f"数据转换接口起始行(Part2-orig): {data_transform_start}")
        
        if data_transform_start > 0:
            boundary_content = orig_part2[boundary_api_start:data_transform_start]
            
            # 添加异常体系（从Part1-original）
            exception_start = find_line_number(orig_part1, '## 4. 异常体系')
            print(f"异常体系起始行(Part1-orig): {exception_start}")
            
            if exception_start > 0:
                exception_content = orig_part1[exception_start:]
                
                part4_content = advanced_postprocess + ["\n---\n\n"] + boundary_content + ["\n---\n\n"] + exception_content
                
                part4_header = create_doc_header(4, "高级后处理、边界条件和异常体系", [1, 2, 3, 5])
                final_part4 = part4_header + part4_content + create_doc_footer(4)
                
                print(f"Part4行数: {len(final_part4)}")
    
    # ===== Part5: 数据转换 + 插件扩展 + 错误码 + 版本管理 =====
    # Part5内容: 从数据转换接口到结尾
    
    if data_transform_start > 0:
        part5_content = orig_part2[data_transform_start:]
        
        part5_header = create_doc_header(5, "数据转换、插件扩展和版本管理", [1, 2, 3, 4])
        
        # 更新内部引用
        updated_part5 = []
        for line in part5_content:
            line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
            line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
            updated_part5.append(line)
        
        final_part5 = part5_header + updated_part5 + create_doc_footer(5, 5)
        
        print(f"Part5行数: {len(final_part5)}")
    
    # ===== 写入文件 =====
    print("\n正在写入最终文件...")
    write_file(base_dir / '2-4_接口文档-Part1.md', final_part1)
    write_file(base_dir / '2-4_接口文档-Part2.md', final_part2)
    write_file(base_dir / '2-4_接口文档-Part3.md', final_part3)
    write_file(base_dir / '2-4_接口文档-Part4.md', final_part4)
    write_file(base_dir / '2-4_接口文档-Part5.md', final_part5)
    
    # 统计结果
    print("\n" + "="*60)
    print("最终文件行数统计:")
    print("="*60)
    files_info = [
        ("Part1", final_part1),
        ("Part2", final_part2),
        ("Part3", final_part3),
        ("Part4", final_part4),
        ("Part5", final_part5),
    ]
    
    all_under_limit = True
    for name, content in files_info:
        line_count = len(content)
        status = "✓" if line_count <= 1000 else "✗ 超过限制"
        if line_count > 1000:
            all_under_limit = False
        print(f"{name}: {line_count:4d} 行 {status}")
    
    print("="*60)
    if all_under_limit:
        print("✓ 所有文件均符合1000行限制！")
    else:
        print("✗ 仍有文件超过限制，需要进一步精简")
    print("="*60)

if __name__ == '__main__':
    main()
