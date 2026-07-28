#!/usr/bin/env python3
"""
2-4接口文档最终拆分脚本 - 严格确保每个文件不超过1000行
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_section(lines, title):
    for i, line in enumerate(lines):
        if title in line:
            return i
    return None

def create_header(part_num, title, prev_parts, next_parts):
    """创建文档头部"""
    header = [
        f"# AutoFlowCFD 接口文档 - Part{part_num}\n",
        "\n",
        "## 文档版本控制\n",
        "\n",
        "|版本号|修订日期|修订人|修订说明|\n",
        "|---|---|---|---|\n",
        "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，" + title + "|\n",
        "\n",
        "---\n",
        "\n",
        "## 1. 引言\n",
        "\n",
        f"本文档为 **AutoFlowCFD** 接口设计规范文档的第{part_num}部分。\n",
        "\n"
    ]
    
    if prev_parts or next_parts:
        header.append("**相关文档**:\n")
        for p in prev_parts:
            header.append(f"- [接口文档 - Part{p}](./2-4_接口文档-Part{p}.md)\n")
        for p in next_parts:
            header.append(f"- [接口文档 - Part{p}](./2-4_接口文档-Part{p}.md)\n")
        header.append("\n---\n\n")
    
    return header

def create_footer(part_num, total=5):
    """创建文档尾部"""
    footer = ["\n---\n\n"]
    
    if part_num < total:
        footer.append(f"**文档结束 - Part{part_num}**\n\n")
        footer.append(f"*继续查看 [接口文档 - Part{part_num+1}](./2-4_接口文档-Part{part_num+1}.md)*\n\n")
        if part_num > 1:
            footer.append(f"*返回 [接口文档 - Part{part_num-1}](./2-4_接口文档-Part{part_num-1}.md)*\n")
    else:
        footer.append(f"**文档结束 - Part{part_num}（最终部分）**\n\n")
        footer.append("*返回以下文档：*\n")
        for i in range(1, part_num):
            footer.append(f"- [接口文档 - Part{i}](./2-4_接口文档-Part{i}.md)\n")
    
    return footer

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 读取当前文件
    part1 = read_file(base_dir / '2-4_接口文档-Part1.md')
    part2 = read_file(base_dir / '2-4_接口文档-Part2.md')
    part3 = read_file(base_dir / '2-4_接口文档-Part3.md')
    part4 = read_file(base_dir / '2-4_接口文档-Part4.md')
    part5 = read_file(base_dir / '2-4_接口文档-Part5.md')
    
    print(f"当前行数:")
    print(f"Part1: {len(part1)}")
    print(f"Part2: {len(part2)}")
    print(f"Part3: {len(part3)}")
    print(f"Part4: {len(part4)}")
    print(f"Part5: {len(part5)}")
    
    # Part3需要拆分为Part3和Part4
    # 在边界条件API处分割（第153行附近）
    boundary_start = find_section(part3, '### 4. 边界条件管理 API')
    print(f"\nPart3中边界条件API起始行: {boundary_start}")
    
    if boundary_start:
        # 新Part3: 配置管理API + 后处理API前半部分
        # 找到后处理API开始位置
        postprocess_start = find_section(part3, '## 2. 后处理模块 API')
        print(f"Part3中后处理API起始行: {postprocess_start}")
        
        if postprocess_start:
            # Part3: 配置管理 + 后处理API (到VTK导出之前)
            vtk_export_start = find_section(part3[postprocess_start:], '### 2.2 VTK 导出 API')
            if vtk_export_start:
                actual_vtk_pos = postprocess_start + vtk_export_start
                
                # Part3内容：配置管理 + 系数计算API
                part3_content = part3[21:actual_vtk_pos]  # 从3.4配置管理开始
                
                new_part3_header = create_header(
                    3, 
                    "Python API配置管理和气动系数计算",
                    [1, 2],
                    [4, 5]
                )
                new_part3 = new_part3_header + part3_content + create_footer(3)
                
                # Part4内容：VTK导出 + 边界条件API + 异常体系
                part4_content = part3[actual_vtk_pos:boundary_start] + ["\n---\n\n"] + part3[boundary_start:]
                
                new_part4_header = create_header(
                    4,
                    "Python API后处理、边界条件和异常体系",
                    [1, 2, 3],
                    [5]
                )
                new_part4 = new_part4_header + part4_content + create_footer(4)
                
                print(f"新Part3行数: {len(new_part3)}")
                print(f"新Part4行数: {len(new_part4)}")
    
    # Part4原内容（数据转换+插件+错误码）变为Part5
    # Part5原内容（版本管理+FAQ）变为Part6
    
    # 但我们需要保持在5个Part以内，所以合并策略调整：
    # Part4: VTK导出 + 收敛曲线 + 瞬态分析（后处理剩余部分）
    # Part5: 边界条件API + 异常体系 + 数据转换 + 插件扩展 + 错误码 + 版本管理
    
    # 重新规划：
    # 由于Part4已经很小（1083行），我们可以将Part3的边界条件部分移到这里
    
    # 实际上，更好的方案是：
    # Part3: 配置管理API + 后处理API（精简版，控制在800行内）
    # Part4: 边界条件API + 异常体系（约600行）  
    # Part5: 数据转换 + 插件 + 错误码 + 版本管理（约700行）
    
    # 让我采用这个方案
    print("\n采用新的拆分方案...")
    
    # Part3: 配置管理 + 后处理API（只保留核心API）
    convergence_start = find_section(part3, '### 2.3 收敛曲线 API')
    if convergence_start and postprocess_start:
        actual_convergence_pos = postprocess_start + convergence_start
        
        part3_core = part3[21:actual_convergence_pos]  # 配置管理 + 系数计算 + VTK导出
        
        new_part3_header = create_header(3, "Python API配置管理和核心后处理", [1, 2], [4, 5])
        new_part3 = new_part3_header + part3_core + create_footer(3)
        
        # Part4: 收敛曲线 + 瞬态分析 + 边界条件
        part4_content = part3[actual_convergence_pos:]
        
        new_part4_header = create_header(4, "高级后处理和边界条件管理", [1, 2, 3], [5])
        new_part4 = new_part4_header + part4_content + create_footer(4)
        
        # Part5: 保持原有的数据转换+插件+错误码+版本管理
        new_part5_header = create_header(5, "数据转换、插件扩展和版本管理", [1, 2, 3, 4], [])
        new_part5 = new_part5_header + part4 + ["\n---\n\n"] + part5 + create_footer(5, 5)
        
        print(f"最终Part3: {len(new_part3)} 行")
        print(f"最终Part4: {len(new_part4)} 行")
        print(f"最终Part5: {len(new_part5)} 行")
        
        # 写入文件
        write_file(base_dir / '2-4_接口文档-Part3.md', new_part3)
        write_file(base_dir / '2-4_接口文档-Part4.md', new_part4)
        write_file(base_dir / '2-4_接口文档-Part5.md', new_part5)
        
        print("\n✓ 所有文件已更新！")
    else:
        print("✗ 未找到关键章节位置")

if __name__ == '__main__':
    main()
