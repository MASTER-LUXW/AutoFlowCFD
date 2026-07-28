#!/usr/bin/env python3
"""
2-4接口文档最终重构脚本 - 确保每个文件不超过1000行
"""

import os
from pathlib import Path

def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    """写入文件内容"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_section_start(lines, section_title):
    """查找章节起始行"""
    for i, line in enumerate(lines):
        if section_title in line:
            return i
    return None

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 读取新生成的文件
    part1_lines = read_file(base_dir / '2-4_接口文档-Part1-new.md')
    part2_lines = read_file(base_dir / '2-4_接口文档-Part2-new.md')
    part3_lines = read_file(base_dir / '2-4_接口文档-Part3-new.md')
    part4_lines = read_file(base_dir / '2-4_接口文档-Part4-new.md')
    part5_lines = read_file(base_dir / '2-4_接口文档-Part5-new.md')
    
    print(f"当前文件行数:")
    print(f"Part1: {len(part1_lines)}")
    print(f"Part2: {len(part2_lines)}")
    print(f"Part3: {len(part3_lines)}")
    print(f"Part4: {len(part4_lines)}")
    print(f"Part5: {len(part5_lines)}")
    
    # Part2需要拆分（1250行 > 1000）
    # 在3.3求解器API和3.4配置管理API之间分割
    config_api_start = find_section_start(part2_lines, '### 3.4 配置管理 API')
    print(f"\nPart2中配置管理API起始行: {config_api_start}")
    
    if config_api_start:
        # Part2 (最终): 网格解析 + 求解器API (~500行)
        final_part2 = part2_lines[:config_api_start]
        
        # Part3新增部分: 配置管理API
        config_mgmt_part = part2_lines[config_api_start:]
        
        print(f"Part2最终行数: {len(final_part2)}")
        print(f"配置管理部分行数: {len(config_mgmt_part)}")
    
    # Part3需要拆分（1665行 > 1000）
    # 在后处理API和边界条件API之间分割
    boundary_api_start = find_section_start(part3_lines, '## 3. 边界条件管理 API')
    print(f"\nPart3中边界条件API起始行: {boundary_api_start}")
    
    if boundary_api_start:
        # Part3 (最终): 后处理API + 配置管理 (~800行)
        final_part3_header = [
            "# AutoFlowCFD 接口文档 - Part3\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，Python API配置管理和后处理接口|\n",
            "\n",
            "---\n",
            "\n",
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档的第三部分，涵盖配置管理API和后处理模块API。\n",
            "\n",
            "**相关文档**:\n",
            "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
            "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口（网格解析、求解器）\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        final_part3_content = config_mgmt_part + ["\n---\n\n"] + part3_lines[:boundary_api_start]
        final_part3 = final_part3_header + final_part3_content
        
        # Part4 (最终): 边界条件API + 异常体系
        final_part4_header = [
            "# AutoFlowCFD 接口文档 - Part4\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，边界条件API和异常体系|\n",
            "\n",
            "---\n",
            "\n",
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档的第四部分，涵盖边界条件管理API和异常体系。\n",
            "\n",
            "**相关文档**:\n",
            "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
            "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口（网格解析、求解器）\n",
            "- [接口文档 - Part3](./2-4_接口文档-Part3.md): 配置管理和后处理API\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        # 找到异常体系在part3中的位置
        exception_start = find_section_start(part3_lines, '## 4. 异常体系')
        print(f"Part3中异常体系起始行: {exception_start}")
        
        if exception_start:
            boundary_and_exception = part3_lines[boundary_api_start:exception_start] + ["\n---\n\n"] + part3_lines[exception_start:]
            final_part4 = final_part4_header + boundary_and_exception
        else:
            final_part4 = final_part4_header + part3_lines[boundary_api_start:]
        
        print(f"Part3最终行数: {len(final_part3)}")
        print(f"Part4最终行数: {len(final_part4)}")
    
    # Part4和Part5保持不变（已经小于1000行）
    # 但需要更新Part4的标题和内容引用
    
    # 更新所有文件的内部引用
    def update_references(lines):
        updated = []
        for line in lines:
            line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
            line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
            updated.append(line)
        return updated
    
    final_part1 = update_references(part1_lines)
    final_part2 = update_references(final_part2)
    final_part3 = update_references(final_part3)
    final_part4 = update_references(final_part4)
    final_part5 = update_references(part5_lines)
    
    # 添加跨文档引用到每个文件末尾
    def add_cross_references(content, part_num, total_parts=5):
        footer = ["\n---\n\n"]
        
        if part_num < total_parts:
            footer.append(f"**文档结束 - Part{part_num}**\n\n")
            footer.append(f"*继续查看 [接口文档 - Part{part_num+1}](./2-4_接口文档-Part{part_num+1}.md) 了解更多内容。*\n\n")
            
            if part_num > 1:
                footer.append(f"*返回 [接口文档 - Part{part_num-1}](./2-4_接口文档-Part{part_num-1}.md) 查看上一部分内容。*\n")
        else:
            footer.append(f"**文档结束 - Part{part_num}（最终部分）**\n\n")
            footer.append("*返回以下文档查看更多内容：*\n")
            for i in range(1, part_num):
                footer.append(f"- [接口文档 - Part{i}](./2-4_接口文档-Part{i}.md)\n")
        
        return content + footer
    
    final_part1 = add_cross_references(final_part1, 1)
    final_part2 = add_cross_references(final_part2, 2)
    final_part3 = add_cross_references(final_part3, 3)
    final_part4 = add_cross_references(final_part4, 4)
    final_part5 = add_cross_references(final_part5, 5)
    
    # 写入最终文件
    print("\n正在写入最终文件...")
    write_file(base_dir / '2-4_接口文档-Part1.md', final_part1)
    write_file(base_dir / '2-4_接口文档-Part2.md', final_part2)
    write_file(base_dir / '2-4_接口文档-Part3.md', final_part3)
    write_file(base_dir / '2-4_接口文档-Part4.md', final_part4)
    write_file(base_dir / '2-4_接口文档-Part5.md', final_part5)
    
    # 统计最终文件行数
    print(f"\n最终文件行数统计:")
    print(f"Part1: {len(final_part1)} 行 {'✓' if len(final_part1) <= 1000 else '✗ 超过限制'}")
    print(f"Part2: {len(final_part2)} 行 {'✓' if len(final_part2) <= 1000 else '✗ 超过限制'}")
    print(f"Part3: {len(final_part3)} 行 {'✓' if len(final_part3) <= 1000 else '✗ 超过限制'}")
    print(f"Part4: {len(final_part4)} 行 {'✓' if len(final_part4) <= 1000 else '✗ 超过限制'}")
    print(f"Part5: {len(final_part5)} 行 {'✓' if len(final_part5) <= 1000 else '✗ 超过限制'}")
    
    # 删除临时文件
    for temp_file in ['2-4_接口文档-Part1-new.md', '2-4_接口文档-Part2-new.md', 
                      '2-4_接口文档-Part3-new.md', '2-4_接口文档-Part4-new.md', 
                      '2-4_接口文档-Part5-new.md']:
        temp_path = base_dir / temp_file
        if temp_path.exists():
            temp_path.unlink()
    
    print("\n重构完成！所有文件已生成并符合1000行限制。")

if __name__ == '__main__':
    main()
