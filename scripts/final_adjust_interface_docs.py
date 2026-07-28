#!/usr/bin/env python3
"""
2-4接口文档最终调整脚本
将Part3的边界条件API移到Part4，确保所有文件不超过1000行
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def create_header(part_num, subtitle):
    """创建文档头部"""
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
        "\n",
        "## 1. 引言\n",
        "\n",
        f"本文档为 **AutoFlowCFD** 接口设计规范文档的第{part_num}部分。\n",
        "\n",
        "**相关文档**:\n",
    ]
    
    related = [i for i in range(1, 6) if i != part_num]
    for p in related:
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
    part3 = read_file(base_dir / '2-4_接口文档-Part3.md')
    part4 = read_file(base_dir / '2-4_接口文档-Part4.md')
    
    print(f"调整前行数:")
    print(f"Part3: {len(part3)}")
    print(f"Part4: {len(part4)}")
    
    # 找到边界条件API起始位置
    boundary_start = None
    for i, line in enumerate(part3):
        if '### 4. 边界条件管理 API' in line:
            boundary_start = i
            break
    
    print(f"\n边界条件API起始行: {boundary_start}")
    
    if boundary_start:
        # Part3新内容：只保留配置管理API
        new_part3_content = part3[:boundary_start]
        
        # Part4新内容：边界条件API + 原有Part4内容
        boundary_content = part3[boundary_start:]
        new_part4_content = boundary_content + ["\n---\n\n"] + part4
        
        # 创建新的头部
        new_part3_header = create_header(3, "Python API配置管理接口")
        new_part4_header = create_header(4, "边界条件管理和异常体系")
        
        # 组装最终内容
        final_part3 = new_part3_header + new_part3_content + create_footer(3)
        final_part4 = new_part4_header + new_part4_content + create_footer(4)
        
        print(f"\n调整后行数:")
        print(f"Part3: {len(final_part3)}")
        print(f"Part4: {len(final_part4)}")
        
        # 写入文件
        write_file(base_dir / '2-4_接口文档-Part3.md', final_part3)
        write_file(base_dir / '2-4_接口文档-Part4.md', final_part4)
        
        print("\n✓ 文件调整完成！")
    
    # 验证所有文件
    print("\n" + "="*60)
    print("最终验证:")
    print("="*60)
    
    all_files = ['Part1', 'Part2', 'Part3', 'Part4', 'Part5']
    all_under_limit = True
    
    for name in all_files:
        file_path = base_dir / f'2-4_接口文档-{name}.md'
        lines = read_file(file_path)
        line_count = len(lines)
        status = "✓" if line_count <= 1000 else "✗ 超过限制"
        if line_count > 1000:
            all_under_limit = False
        print(f"{name}: {line_count:4d} 行 {status}")
    
    print("="*60)
    if all_under_limit:
        print("✓✓✓ 所有文件均符合1000行限制！重构成功！")
    else:
        print("✗ 仍有文件超过限制")
    print("="*60)

if __name__ == '__main__':
    main()
