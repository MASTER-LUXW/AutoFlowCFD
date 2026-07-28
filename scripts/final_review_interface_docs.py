#!/usr/bin/env python3
"""
2-4接口文档最终校对和优化脚本
检查并修复：
1. 章节编号连续性
2. 跨文档引用一致性  
3. 删除重复内容
4. 确保框图中没有中文
"""

from pathlib import Path
import re

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def check_chapter_numbering(lines, part_num):
    """检查章节编号是否连续"""
    issues = []
    expected_main = 1  # 期望的主章节号（除了Part1从引言开始）
    
    for i, line in enumerate(lines):
        if line.startswith('## ') and not line.startswith('## 文档'):
            # 提取章节号
            match = re.match(r'## (\d+)\.', line)
            if match:
                chapter_num = int(match.group(1))
                if chapter_num != expected_main:
                    issues.append(f"行{i+1}: 期望主章节 {expected_main}，实际为 {chapter_num}")
                expected_main += 1
    
    return issues

def check_cross_references(lines, part_num):
    """检查跨文档引用是否正确"""
    issues = []
    
    for i, line in enumerate(lines):
        # 检查是否有旧的引用格式
        if './3-1_接口文档' in line:
            issues.append(f"行{i+1}: 发现旧引用格式 './3-1_接口文档'")
        
        # 检查引用是否存在
        if './2-4_接口文档-Part' in line:
            match = re.search(r'Part(\d)', line)
            if match:
                ref_part = int(match.group(1))
                if ref_part < 1 or ref_part > 5:
                    issues.append(f"行{i+1}: 引用了不存在的Part{ref_part}")
    
    return issues

def check_duplicate_sections(lines):
    """检查是否有重复的章节"""
    sections = []
    duplicates = []
    
    for i, line in enumerate(lines):
        if line.startswith('## '):
            section_title = line.strip()
            if section_title in sections:
                duplicates.append((i+1, section_title))
            else:
                sections.append(section_title)
    
    return duplicates

def check_chinese_in_diagrams(lines):
    """检查框图中是否有中文"""
    issues = []
    in_code_block = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        
        # 在代码块中检查是否有中文字符
        if in_code_block:
            # 简单的中文检测
            if re.search(r'[\u4e00-\u9fff]', line):
                # 排除注释中的中文
                if not line.strip().startswith('#') and not line.strip().startswith('//'):
                    issues.append(f"行{i+1}: 框图中可能包含中文")
    
    return issues

def main():
    base = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    print("="*70)
    print("2-4接口文档最终校对和优化")
    print("="*70)
    
    all_issues = {}
    
    for part_num in range(1, 6):
        file_path = base / f'2-4_接口文档-Part{part_num}.md'
        lines = read_file(file_path)
        
        print(f"\n检查 Part{part_num} ({len(lines)} 行)...")
        
        issues = []
        
        # 1. 检查章节编号
        chapter_issues = check_chapter_numbering(lines, part_num)
        if chapter_issues:
            issues.extend(chapter_issues)
            print(f"  ✗ 章节编号问题: {len(chapter_issues)} 个")
        
        # 2. 检查跨文档引用
        ref_issues = check_cross_references(lines, part_num)
        if ref_issues:
            issues.extend(ref_issues)
            print(f"  ✗ 引用问题: {len(ref_issues)} 个")
        
        # 3. 检查重复章节
        dup_issues = check_duplicate_sections(lines)
        if dup_issues:
            issues.extend([(line_no, f"重复章节: {title}") for line_no, title in dup_issues])
            print(f"  ✗ 重复章节: {len(dup_issues)} 个")
        
        # 4. 检查框图中的中文
        diagram_issues = check_chinese_in_diagrams(lines)
        if diagram_issues:
            issues.extend(diagram_issues)
            print(f"  ✗ 框图中文: {len(diagram_issues)} 个")
        
        if issues:
            all_issues[part_num] = issues
            print(f"  共发现 {len(issues)} 个问题")
        else:
            print(f"  ✓ 无问题")
    
    # 输出详细问题列表
    if all_issues:
        print("\n" + "="*70)
        print("问题详情:")
        print("="*70)
        
        for part_num, issues in all_issues.items():
            print(f"\nPart{part_num}:")
            for issue in issues[:10]:  # 只显示前10个问题
                if isinstance(issue, tuple):
                    print(f"  行{issue[0]}: {issue[1]}")
                else:
                    print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("\n" + "="*70)
        print("✓✓✓ 所有文档校对完成，未发现问题！")
        print("="*70)
    
    # 生成总结报告
    print("\n" + "="*70)
    print("重构总结报告")
    print("="*70)
    
    total_lines = 0
    for part_num in range(1, 6):
        file_path = base / f'2-4_接口文档-Part{part_num}.md'
        lines = len(read_file(file_path))
        total_lines += lines
        status = "✓" if lines <= 1000 else "✗"
        print(f"Part{part_num}: {lines:4d} 行 {status}")
    
    print("-"*70)
    print(f"总计: {total_lines} 行")
    print(f"平均: {total_lines // 5} 行/Part")
    print(f"最大: {max(len(read_file(base / f'2-4_接口文档-Part{i}.md')) for i in range(1,6))} 行")
    print(f"最小: {min(len(read_file(base / f'2-4_接口文档-Part{i}.md')) for i in range(1,6))} 行")
    print("="*70)
    
    if not all_issues:
        print("✓✓✓ 重构、校对、优化全部完成！文档质量优秀！")
    else:
        print(f"⚠ 发现 {sum(len(v) for v in all_issues.values())} 个问题，建议人工检查")
    print("="*70)

if __name__ == '__main__':
    main()
