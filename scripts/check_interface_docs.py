#!/usr/bin/env python3
"""检查2-4接口文档各Part的行数"""

from pathlib import Path

base = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')

print("="*60)
print("2-4接口文档行数检查")
print("="*60)

all_ok = True
for i in range(1, 6):
    file_path = base / f'2-4_接口文档-Part{i}.md'
    if file_path.exists():
        lines = len(open(file_path, 'r', encoding='utf-8').readlines())
        status = "✓" if lines <= 1000 else "✗ 超过限制"
        if lines > 1000:
            all_ok = False
        print(f"Part{i}: {lines:4d} 行 {status}")
    else:
        print(f"Part{i}: 文件不存在")
        all_ok = False

print("="*60)
if all_ok:
    print("✓ 所有文件均符合1000行限制！")
else:
    print("✗ 存在超过1000行的文件")
print("="*60)
