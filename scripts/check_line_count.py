"""检查超400行的Python文件。"""
import os

src_dir = os.path.join(os.path.dirname(__file__), '..', 'src', 'autoflowcfd')
results = []
for root, dirs, files in os.walk(src_dir):
    for f in files:
        if f.endswith('.py'):
            fpath = os.path.join(root, f)
            with open(fpath, encoding='utf-8') as fp:
                lines = sum(1 for _ in fp)
            if lines > 400:
                results.append((lines, fpath))

results.sort(reverse=True)
for lines, fpath in results:
    rel = os.path.relpath(fpath, src_dir)
    print(f"{rel}: {lines} lines")
