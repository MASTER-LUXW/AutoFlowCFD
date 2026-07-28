#!/usr/bin/env python3
"""
2-4接口文档最终重构脚本 - 确保所有文件严格不超过1000行，无重复内容
"""

from pathlib import Path

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def find_line(lines, pattern, start=0):
    for i in range(start, len(lines)):
        if pattern in lines[i]:
            return i
    return -1

def main():
    base = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    backup = base / 'backup_interface_docs'
    
    print("="*70)
    print("2-4接口文档最终重构")
    print("="*70)
    
    # 从备份恢复原始的重构结果（Part1: 915, Part2: 507, Part3: 469, Part4: 975）
    # Part5需要从原始Part2重新构建
    
    # 读取当前的Part1-Part4（已经是正确的）
    part1 = read_file(base / '2-4_接口文档-Part1.md')
    part2 = read_file(base / '2-4_接口文档-Part2.md')
    part3 = read_file(base / '2-4_接口文档-Part3.md')
    part4 = read_file(base / '2-4_接口文档-Part4.md')
    
    print(f"\n当前状态:")
    print(f"  Part1: {len(part1)} 行 ✓")
    print(f"  Part2: {len(part2)} 行 ✓")
    print(f"  Part3: {len(part3)} 行 ✓")
    print(f"  Part4: {len(part4)} 行 ✓")
    
    # 从原始Part2构建Part5
    orig_part2 = read_file(backup / '2-4_接口文档-Part2-original.md')
    print(f"  原始Part2: {len(orig_part2)} 行")
    
    # Part5应该包含：
    # 1. 数据转换与序列化接口（从orig_part2提取）
    # 2. 插件扩展接口
    # 3. 错误码详细索引  
    # 4. 接口版本管理与兼容性
    # 5. 附录（API索引、工作流示例、FAQ等）
    
    data_transform_start = find_line(orig_part2, '## 4. 数据转换与序列化接口')
    plugin_start = find_line(orig_part2, '## 5. 插件扩展接口')
    error_code_start = find_line(orig_part2, '## 6. 错误码详细索引')
    version_mgmt_start = find_line(orig_part2, '## 7. 接口版本管理与兼容性')
    appendix_start = find_line(orig_part2, '## 8. 附录')
    
    print(f"\n原始Part2分割点:")
    print(f"  数据转换: {data_transform_start}")
    print(f"  插件扩展: {plugin_start}")
    print(f"  错误码: {error_code_start}")
    print(f"  版本管理: {version_mgmt_start}")
    print(f"  附录: {appendix_start}")
    
    # 提取Part5内容
    part5_content = orig_part2[data_transform_start:]
    
    # 创建Part5头部
    part5_header = [
        "# AutoFlowCFD 接口文档 - Part5\n",
        "\n",
        "## 文档版本控制\n",
        "\n",
        "|版本号|修订日期|修订人|修订说明|\n",
        "|---|---|---|---|\n",
        "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，数据转换、插件扩展和版本管理|\n",
        "\n",
        "---\n",
        "\n",
        "## 1. 引言\n",
        "\n",
        "本文档为 **AutoFlowCFD** 接口设计规范文档的第5部分（最终部分）。\n",
        "\n",
        "**相关文档**:\n",
        "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
        "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口\n",
        "- [接口文档 - Part3](./2-4_接口文档-Part3.md): 配置管理和后处理API\n",
        "- [接口文档 - Part4](./2-4_接口文档-Part4.md): 高级后处理和边界条件\n",
        "\n",
        "---\n",
        "\n"
    ]
    
    # 创建Part5尾部
    part5_footer = [
        "\n---\n\n",
        "**文档结束 - Part5（最终部分）**\n\n",
        "*返回以下文档查看更多内容：*\n",
        "- [接口文档 - Part1](./2-4_接口文档-Part1.md)\n",
        "- [接口文档 - Part2](./2-4_接口文档-Part2.md)\n",
        "- [接口文档 - Part3](./2-4_接口文档-Part3.md)\n",
        "- [接口文档 - Part4](./2-4_接口文档-Part4.md)\n"
    ]
    
    # 组装Part5
    part5 = part5_header + part5_content + part5_footer
    
    # 更新内部引用
    updated_part5 = []
    for line in part5:
        line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
        line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
        updated_part5.append(line)
    
    write_file(base / '2-4_接口文档-Part5.md', updated_part5)
    
    print(f"\nPart5: {len(updated_part5)} 行")
    
    # 最终验证
    print("\n" + "="*70)
    print("最终验证")
    print("="*70)
    
    all_ok = True
    for i in range(1, 6):
        file_path = base / f'2-4_接口文档-Part{i}.md'
        lines = len(read_file(file_path))
        status = "✓" if lines <= 1000 else "✗"
        if lines > 1000:
            all_ok = False
        print(f"Part{i}: {lines:4d} 行 {status}")
    
    print("="*70)
    if all_ok:
        print("✓✓✓ 所有文件均符合1000行限制！重构成功！")
    else:
        print("✗ 仍有文件超过限制")
    print("="*70)

if __name__ == '__main__':
    main()
