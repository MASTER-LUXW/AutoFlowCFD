#!/usr/bin/env python3
"""
2-4接口文档最终精简脚本
精简Part1和Part3，确保所有文件不超过1000行
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def compact_section(lines, section_pattern, max_examples=2):
    """精简某个章节的示例数量"""
    result = []
    example_count = 0
    skip_block = False
    
    for line in lines:
        if section_pattern in line:
            result.append(line)
            continue
        
        if '**使用示例**' in line or '**Use Example**' in line:
            example_count += 1
            if example_count > max_examples:
                result.append('\n*更多示例请参考官方文档或CLI帮助。*\n\n')
                skip_block = True
                continue
        
        if skip_block:
            if line.strip().startswith('```'):
                skip_block = False
            continue
        
        result.append(line)
    
    return result

def remove_redundant_json_blocks(lines):
    """删除冗余的JSON输出格式块，保留一个模板"""
    result = []
    json_block_count = 0
    skip_json = False
    
    for line in lines:
        if '**JSON 输出格式**' in line:
            json_block_count += 1
            if json_block_count == 1:
                result.append(line)
            else:
                result.append('\n*JSON输出格式与首个命令类似，包含command、status、timestamp、result字段。*\n\n')
                skip_json = True
                continue
        
        if skip_json:
            if line.strip().startswith('```json'):
                continue
            elif line.strip() == '```':
                skip_json = False
                continue
            else:
                continue
        
        result.append(line)
    
    return result

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 处理Part1 (1051行 -> 目标 < 1000)
    print("精简Part1...")
    part1 = read_file(base_dir / '2-4_接口文档-Part1.md')
    
    # 策略1: 减少每个CLI命令的示例数量（从多个减到2个）
    part1_compact = compact_section(part1, 'CLI-', max_examples=2)
    
    # 策略2: 删除冗余的JSON输出块
    part1_compact = remove_redundant_json_blocks(part1_compact)
    
    # 策略3: 精简日志级别说明
    simplified_part1 = []
    in_log_section = False
    for line in part1_compact:
        if '### 2.9 日志级别规范' in line:
            in_log_section = True
            simplified_part1.append(line)
        elif in_log_section and '|ERROR|' in line:
            simplified_part1.append(line)
            simplified_part1.append('|WARNING|警告信息，程序继续|`WARNING: High aspect ratio cells detected`|\n')
            simplified_part1.append('|INFO|常规进度信息|`INFO: Iteration 1000, Res=1.2e-4, Cd=0.289`|\n')
            simplified_part1.append('|DEBUG|调试详情|`DEBUG: CFL adjusted from 1.5 to 1.8`|\n')
            # 跳过原有的详细表格
            in_log_section = False
        elif in_log_section and (line.startswith('|') or line.startswith('-')):
            continue
        else:
            simplified_part1.append(line)
    
    part1_final = simplified_part1
    
    # 如果还是超过1000行，进一步精简
    if len(part1_final) > 1000:
        print(f"  Part1仍为 {len(part1_final)} 行，进一步精简...")
        # 删除详细的错误处理示例
        final_part1 = []
        error_example_count = 0
        skip_error_block = False
        
        for line in part1_final:
            if '**错误处理**' in line:
                error_example_count += 1
                if error_example_count > 1:
                    final_part1.append('\n*错误处理模式类似，根据退出码判断错误类型。*\n\n')
                    skip_error_block = True
                    continue
            
            if skip_error_block:
                if line.strip().startswith('```bash'):
                    continue
                elif line.strip() == '```':
                    skip_error_block = False
                    continue
                else:
                    continue
            
            final_part1.append(line)
        
        part1_final = final_part1
    
    write_file(base_dir / '2-4_接口文档-Part1.md', part1_final)
    print(f"Part1最终行数: {len(part1_final)}")
    
    # 处理Part3 (1057行 -> 目标 < 1000)
    print("\n精简Part3...")
    part3 = read_file(base_dir / '2-4_接口文档-Part3.md')
    
    # 策略1: 减少配置管理API的示例
    part3_compact = compact_section(part3, 'API-CONFIG-', max_examples=1)
    
    # 策略2: 精简后处理API的详细docstring
    simplified_part3 = []
    api_detail_count = 0
    skip_api_detail = False
    
    for line in part3_compact:
        if '#### API-POST-' in line:
            api_detail_count += 1
            if api_detail_count > 2:  # 只详细说明前2个后处理API
                simplified_part3.append(line.split(':')[0] + ': *详见官方API参考文档*\n\n')
                skip_api_detail = True
                continue
        
        if skip_api_detail:
            if line.startswith('### ') or line.startswith('## '):
                skip_api_detail = False
                simplified_part3.append(line)
            elif line.startswith('#### '):
                skip_api_detail = False
                simplified_part3.append(line)
            else:
                continue
        else:
            simplified_part3.append(line)
    
    part3_final = simplified_part3
    
    # 如果还是超过1000行，进一步精简
    if len(part3_final) > 1000:
        print(f"  Part3仍为 {len(part3_final)} 行，进一步精简...")
        # 删除配置管理的部分参数说明表格
        final_part3 = []
        param_table_count = 0
        
        for line in part3_final:
            if '|选项|短选项|类型|默认值|描述|' in line:
                param_table_count += 1
                if param_table_count > 3:  # 只保留前3个参数表格
                    final_part3.append('\n*完整参数列表请参考YAML配置文件模板。*\n\n')
                    # 跳过整个表格
                    while final_part3 and not final_part3[-1].strip().startswith('|---'):
                        final_part3.pop()
                    continue
            
            final_part3.append(line)
        
        part3_final = final_part3
    
    write_file(base_dir / '2-4_接口文档-Part3.md', part3_final)
    print(f"Part3最终行数: {len(part3_final)}")
    
    # 统计最终结果
    print("\n" + "="*60)
    print("最终验证:")
    print("="*60)
    
    all_files = {
        'Part1': part1_final,
        'Part2': read_file(base_dir / '2-4_接口文档-Part2.md'),
        'Part3': part3_final,
        'Part4': read_file(base_dir / '2-4_接口文档-Part4.md'),
        'Part5': read_file(base_dir / '2-4_接口文档-Part5.md'),
    }
    
    all_under_limit = True
    for name, content in all_files.items():
        line_count = len(content)
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
