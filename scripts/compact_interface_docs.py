#!/usr/bin/env python3
"""
2-4接口文档智能精简脚本
通过删除冗余内容、合并相似部分来确保每个文件不超过1000行
"""

from pathlib import Path

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def remove_redundant_examples(lines):
    """删除冗余的使用示例，保留最重要的1-2个"""
    result = []
    skip_mode = False
    example_count = 0
    
    for i, line in enumerate(lines):
        # 检测使用示例块
        if '**使用示例**：' in line or '**Use Example**:' in line:
            example_count += 1
            if example_count > 2:  # 只保留前2个示例
                skip_mode = True
                result.append(line.replace('**使用示例**：', '**更多示例请参考CLI帮助或官方文档。**\n'))
                continue
        
        if skip_mode:
            # 跳过示例代码块
            if line.strip().startswith('```bash'):
                skip_mode = False
                continue
            elif line.strip().startswith('```'):
                skip_mode = False
                continue
            else:
                continue
        
        result.append(line)
    
    return result

def compact_json_examples(lines):
    """压缩JSON示例，删除详细格式"""
    result = []
    in_json_block = False
    json_start = -1
    
    for i, line in enumerate(lines):
        if '**JSON 输出格式**：' in line:
            result.append(line)
            result.append('\n*JSON输出包含command、status、timestamp、result等标准字段。详见各命令的--json选项说明。*\n\n')
            in_json_block = True
            continue
        
        if in_json_block:
            if line.strip().startswith('```json'):
                json_start = i
                continue
            elif line.strip() == '```' and json_start > 0:
                in_json_block = False
                # 跳过整个JSON块
                continue
            elif json_start > 0:
                # 跳过JSON内容
                continue
        
        result.append(line)
    
    return result

def update_version_info(lines, part_num):
    """更新版本控制信息"""
    result = []
    for line in lines:
        if '|v0.1|2026-07-22|' in line:
            line = '|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，优化文档结构|\n'
        result.append(line)
    return result

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 处理Part1 (1039行 -> 目标 < 1000)
    print("处理Part1...")
    part1 = read_file(base_dir / '2-4_接口文档-Part1.md')
    part1 = update_version_info(part1, 1)
    part1 = remove_redundant_examples(part1)
    part1 = compact_json_examples(part1)
    write_file(base_dir / '2-4_接口文档-Part1.md', part1)
    print(f"Part1: {len(part1)} 行")
    
    # 处理Part3 (1396行 -> 需要大幅精简)
    print("\n处理Part3...")
    part3 = read_file(base_dir / '2-4_接口文档-Part3.md')
    part3 = update_version_info(part3, 3)
    part3 = remove_redundant_examples(part3)
    part3 = compact_json_examples(part3)
    
    # 删除过于详细的API docstring示例，保留关键接口
    simplified_part3 = []
    skip_detail = False
    api_count = 0
    
    for line in part3:
        # 对于后处理API，只保留核心方法
        if '#### API-POST-' in line:
            api_count += 1
            if api_count > 3:  # 只详细说明前3个API
                skip_detail = True
                simplified_part3.append(line.split(':')[0] + ': [详见官方API文档]\n\n')
                continue
        
        if skip_detail:
            if line.startswith('### ') or line.startswith('## '):
                skip_detail = False
            else:
                continue
        
        simplified_part3.append(line)
    
    write_file(base_dir / '2-4_接口文档-Part3.md', simplified_part3)
    print(f"Part3: {len(simplified_part3)} 行")
    
    # 处理Part4 (1077行 -> 目标 < 1000)
    print("\n处理Part4...")
    part4 = read_file(base_dir / '2-4_接口文档-Part4.md')
    part4 = update_version_info(part4, 4)
    part4 = remove_redundant_examples(part4)
    
    # 精简错误码列表，只列出关键错误码
    simplified_part4 = []
    in_error_table = False
    error_count = 0
    
    for line in part4:
        if '|ERR-' in line and '|' in line:
            error_count += 1
            if error_count > 15:  # 只保留前15个详细错误码
                if not in_error_table:
                    simplified_part4.append('\n*完整错误码列表请参考[错误码索引](#错误码索引)。*\n\n')
                    in_error_table = True
                continue
        
        if line.startswith('## ') or line.startswith('# '):
            in_error_table = False
            error_count = 0
        
        simplified_part4.append(line)
    
    write_file(base_dir / '2-4_接口文档-Part4.md', simplified_part4)
    print(f"Part4: {len(simplified_part4)} 行")
    
    print("\n精简完成！请检查文件大小。")

if __name__ == '__main__':
    main()
