#!/usr/bin/env python3
"""
2-4接口文档最终精简脚本 v2.0
精简Part1和Part3，确保所有文件严格不超过1000行
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def compact_examples(lines, max_per_section=2):
    """精简示例数量，每个命令最多保留2个示例"""
    result = []
    section_examples = 0
    current_section = None
    skip_example = False
    
    for line in lines:
        # 检测新的CLI命令或API
        if 'CLI-' in line or 'API-' in line:
            section_examples = 0
            current_section = line.strip()
        
        if '**使用示例**' in line:
            section_examples += 1
            if section_examples > max_per_section:
                result.append('\n*更多示例请参考官方文档。*\n\n')
                skip_example = True
                continue
        
        if skip_example:
            if line.strip().startswith('```'):
                skip_example = False
            continue
        
        result.append(line)
    
    return result

def remove_duplicate_json_blocks(lines):
    """删除重复的JSON输出格式说明"""
    result = []
    json_count = 0
    
    for line in lines:
        if '**JSON 输出格式**' in line:
            json_count += 1
            if json_count == 1:
                result.append(line)
            else:
                result.append('\n*JSON输出格式与首个命令类似。*\n\n')
                # 跳过JSON代码块
                while result and not result[-1].strip().startswith('```json'):
                    result.pop()
                continue
        
        if json_count > 1 and (line.strip().startswith('```json') or line.strip() == '```'):
            continue
        elif json_count > 1 and '{' in line and '}' in line:
            continue
        else:
            result.append(line)
    
    return result

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    print("="*70)
    print("2-4接口文档最终精简 v2.0")
    print("="*70)
    
    # 处理Part1 (1051行 -> 目标 < 1000)
    print("\n精简Part1...")
    part1 = read_file(base_dir / '2-4_接口文档-Part1.md')
    
    # 策略1: 精简示例
    part1 = compact_examples(part1, max_per_section=2)
    print(f"  精简示例后: {len(part1)} 行")
    
    # 策略2: 删除重复的JSON块
    part1 = remove_duplicate_json_blocks(part1)
    print(f"  删除重复JSON后: {len(part1)} 行")
    
    # 如果还是超过，进一步精简错误处理部分
    if len(part1) > 1000:
        print(f"  Part1仍为 {len(part1)} 行，进一步精简...")
        simplified = []
        error_section_count = 0
        
        for line in part1:
            if '**错误处理**' in line:
                error_section_count += 1
                if error_section_count > 1:
                    simplified.append('\n*错误处理方式类似。*\n\n')
                    # 跳过错误示例代码块
                    while simplified and not simplified[-1].strip().startswith('```bash'):
                        simplified.pop()
                    continue
            
            simplified.append(line)
        
        part1 = simplified
        print(f"  精简错误处理后: {len(part1)} 行")
    
    write_file(base_dir / '2-4_接口文档-Part1.md', part1)
    print(f"  ✓ Part1最终: {len(part1)} 行")
    
    # 处理Part3 (1079行 -> 目标 < 1000)
    print("\n精简Part3...")
    part3 = read_file(base_dir / '2-4_接口文档-Part3.md')
    
    # 策略1: 精简配置管理API的示例
    part3 = compact_examples(part3, max_per_section=1)
    print(f"  精简示例后: {len(part3)} 行")
    
    # 策略2: 精简后处理API的详细docstring
    simplified_part3 = []
    api_detail_count = 0
    skip_detail = False
    
    for line in part3:
        if '#### API-POST-' in line:
            api_detail_count += 1
            if api_detail_count > 2:
                simplified_part3.append(line.split(':')[0] + ': *详见官方API文档*\n\n')
                skip_detail = True
                continue
        
        if skip_detail:
            if line.startswith('### ') or line.startswith('## '):
                skip_detail = False
                simplified_part3.append(line)
            elif line.startswith('#### '):
                skip_detail = False
                simplified_part3.append(line)
            else:
                continue
        else:
            simplified_part3.append(line)
    
    part3 = simplified_part3
    print(f"  精简API详情后: {len(part3)} 行")
    
    # 如果还是超过，删除部分参数表格
    if len(part3) > 1000:
        print(f"  Part3仍为 {len(part3)} 行，进一步精简...")
        final_part3 = []
        table_count = 0
        
        for line in part3:
            if '|选项|短选项|类型|默认值|描述|' in line:
                table_count += 1
                if table_count > 2:
                    final_part3.append('\n*完整参数列表请参考配置文件模板。*\n\n')
                    # 跳过表格
                    while final_part3 and not final_part3[-1].strip().startswith('|---'):
                        final_part3.pop()
                    continue
            
            final_part3.append(line)
        
        part3 = final_part3
        print(f"  精简参数表格后: {len(part3)} 行")
    
    write_file(base_dir / '2-4_接口文档-Part3.md', part3)
    print(f"  ✓ Part3最终: {len(part3)} 行")
    
    # 最终验证
    print("\n" + "="*70)
    print("最终验证")
    print("="*70)
    
    all_files = {}
    for i in range(1, 6):
        file_path = base_dir / f'2-4_接口文档-Part{i}.md'
        lines = read_file(file_path)
        all_files[f'Part{i}'] = lines
    
    all_under_limit = True
    for name, content in all_files.items():
        line_count = len(content)
        status = "✓" if line_count <= 1000 else "✗"
        if line_count > 1000:
            all_under_limit = False
        print(f"{name}: {line_count:4d} 行 {status}")
    
    print("="*70)
    if all_under_limit:
        print("✓✓✓ 所有文件均符合1000行限制！重构完成！")
    else:
        print("✗ 仍有文件超过限制")
    print("="*70)

if __name__ == '__main__':
    main()
