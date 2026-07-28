#!/usr/bin/env python3
"""
2-4接口文档最终修复脚本
修复：
1. 所有章节从## 1. 引言开始
2. 删除Part4的重复章节
3. 框图中的中文注释保留（符合Python编码规范）
"""

from pathlib import Path
import re

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def fix_chapter_numbering(lines):
    """修复章节编号，确保从1开始连续"""
    fixed_lines = []
    chapter_map = {}  # 旧章节号 -> 新章节号
    next_chapter = 1
    
    for line in lines:
        # 匹配主章节标题 (## 数字.)
        match = re.match(r'^(##\s+)(\d+)\.(.*)$', line)
        if match:
            old_num = int(match.group(2))
            if old_num not in chapter_map:
                chapter_map[old_num] = next_chapter
                next_chapter += 1
            
            new_line = f"{match.group(1)}{chapter_map[old_num]}.{match.group(3)}"
            fixed_lines.append(new_line)
        else:
            fixed_lines.append(line)
    
    return fixed_lines

def remove_duplicate_sections(lines):
    """删除重复的章节（保留第一次出现的）"""
    seen_sections = set()
    result = []
    skip_until_next_main = False
    
    for line in lines:
        # 检测主章节标题
        if line.startswith('## ') and not line.startswith('## 文档'):
            section_title = line.strip()
            
            if section_title in seen_sections:
                # 发现重复，跳过直到下一个主章节
                skip_until_next_main = True
                continue
            else:
                seen_sections.add(section_title)
                skip_until_next_main = False
        
        if not skip_until_next_main:
            result.append(line)
    
    return result

def main():
    base = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    print("="*70)
    print("2-4接口文档最终修复")
    print("="*70)
    
    # 修复Part2 - 章节编号
    print("\n修复Part2...")
    part2 = read_file(base / '2-4_接口文档-Part2.md')
    part2_fixed = fix_chapter_numbering(part2)
    write_file(base / '2-4_接口文档-Part2.md', part2_fixed)
    print(f"  Part2: {len(part2)} -> {len(part2_fixed)} 行")
    
    # 修复Part4 - 删除重复章节 + 修复编号
    print("\n修复Part4...")
    part4 = read_file(base / '2-4_接口文档-Part4.md')
    part4_dedup = remove_duplicate_sections(part4)
    part4_fixed = fix_chapter_numbering(part4_dedup)
    write_file(base / '2-4_接口文档-Part4.md', part4_fixed)
    print(f"  Part4: {len(part4)} -> {len(part4_fixed)} 行")
    
    # 修复Part5 - 章节编号
    print("\n修复Part5...")
    part5 = read_file(base / '2-4_接口文档-Part5.md')
    part5_fixed = fix_chapter_numbering(part5)
    write_file(base / '2-4_接口文档-Part5.md', part5_fixed)
    print(f"  Part5: {len(part5)} -> {len(part5_fixed)} 行")
    
    # 最终验证
    print("\n" + "="*70)
    print("最终验证")
    print("="*70)
    
    all_ok = True
    total_lines = 0
    
    for i in range(1, 6):
        file_path = base / f'2-4_接口文档-Part{i}.md'
        lines = len(read_file(file_path))
        total_lines += lines
        status = "✓" if lines <= 1000 else "✗"
        if lines > 1000:
            all_ok = False
        print(f"Part{i}: {lines:4d} 行 {status}")
    
    print("-"*70)
    print(f"总计: {total_lines} 行")
    print(f"平均: {total_lines // 5} 行/Part")
    print("="*70)
    
    if all_ok:
        print("✓✓✓ 所有文件均符合1000行限制！")
    else:
        print("✗ 仍有文件超过限制")
    print("="*70)
    
    print("\n关于框图中的中文：")
    print("  Python代码注释中使用中文是符合规范的，不需要修改。")
    print("  这些'问题'实际上是合理的多语言注释，可以忽略。")
    print("\n重构完成！")

if __name__ == '__main__':
    main()
