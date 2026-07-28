#!/usr/bin/env python3
"""使用原始Part2.backup重建Part5"""

from pathlib import Path

base = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')

# 读取原始Part2.backup
orig_part2 = open(base / '2-4_接口文档-Part2.md.backup', 'r', encoding='utf-8').readlines()
print(f'原始Part2.backup: {len(orig_part2)} 行\n')

# 找到数据转换接口起始位置
data_transform_start = None
for i, line in enumerate(orig_part2):
    if '## 4. 数据转换与序列化接口' in line:
        data_transform_start = i
        break

print(f'数据转换接口起始行: {data_transform_start}')
print(f'Part5内容行数: {len(orig_part2) - data_transform_start}')

# 提取Part5内容
part5_content = orig_part2[data_transform_start:]

# 创建头部
part5_header = [
    '# AutoFlowCFD 接口文档 - Part5\n',
    '\n',
    '## 文档版本控制\n',
    '\n',
    '|版本号|修订日期|修订人|修订说明|\n',
    '|---|---|---|---|\n',
    '|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，数据转换、插件扩展和版本管理|\n',
    '\n',
    '---\n',
    '\n',
    '## 1. 引言\n',
    '\n',
    '本文档为 **AutoFlowCFD** 接口设计规范文档的第5部分（最终部分）。\n',
    '\n',
    '**相关文档**:\n',
    '- [接口文档 - Part1](./2-4_接口文档-Part1.md)\n',
    '- [接口文档 - Part2](./2-4_接口文档-Part2.md)\n',
    '- [接口文档 - Part3](./2-4_接口文档-Part3.md)\n',
    '- [接口文档 - Part4](./2-4_接口文档-Part4.md)\n',
    '\n',
    '---\n',
    '\n'
]

# 创建尾部
part5_footer = [
    '\n---\n\n',
    '**文档结束 - Part5（最终部分）**\n\n',
    '*返回以下文档：*\n',
    '- [接口文档 - Part1](./2-4_接口文档-Part1.md)\n',
    '- [接口文档 - Part2](./2-4_接口文档-Part2.md)\n',
    '- [接口文档 - Part3](./2-4_接口文档-Part3.md)\n',
    '- [接口文档 - Part4](./2-4_接口文档-Part4.md)\n'
]

# 组装
part5 = part5_header + part5_content + part5_footer

# 更新引用
updated = []
for line in part5:
    line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
    line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
    updated.append(line)

# 写入
with open(base / '2-4_接口文档-Part5.md', 'w', encoding='utf-8') as f:
    f.writelines(updated)

print(f'\nPart5已重建: {len(updated)} 行')

# 验证所有文件
print('\n最终验证:')
all_ok = True
for i in range(1, 6):
    lines = len(open(base / f'2-4_接口文档-Part{i}.md', 'r', encoding='utf-8').readlines())
    status = '✓' if lines <= 1000 else '✗'
    if lines > 1000:
        all_ok = False
    print(f'Part{i}: {lines:4d} 行 {status}')

print('\n' + '='*60)
if all_ok:
    print('✓✓✓ 所有文件均符合1000行限制！重构完成！')
else:
    print('✗ 仍有文件超过限制')
print('='*60)
