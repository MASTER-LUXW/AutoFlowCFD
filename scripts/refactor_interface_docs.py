#!/usr/bin/env python3
"""
2-4接口文档重构脚本
将原有的Part1和Part2拆分为5个文档，每个不超过1000行
"""

import os
from pathlib import Path

def read_file(file_path):
    """读取文件内容"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()

def write_file(file_path, lines):
    """写入文件内容"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def count_lines(lines):
    """统计非空行数"""
    return len([line for line in lines if line.strip()])

def main():
    base_dir = Path('d:/myWorkspace/AutoFlowCFD/ProjectFiles')
    
    # 读取原始文件
    part1_lines = read_file(base_dir / '2-4_接口文档-Part1.md')
    part2_lines = read_file(base_dir / '2-4_接口文档-Part2.md')
    
    print(f"原始Part1行数: {len(part1_lines)}")
    print(f"原始Part2行数: {len(part2_lines)}")
    
    # 拆分策略：
    # Part1 (新): 引言 + CLI接口规范 (第1-1036行)
    # Part2 (新): Python API - 网格+求解器+配置 (从原Part1提取)
    # Part3 (新): Python API - 后处理+边界条件+异常 (从原Part1+Part2提取)
    # Part4 (新): 数据转换+插件扩展+错误码 (从原Part2提取)
    # Part5 (新): 版本管理+FAQ+附录 (从原Part2提取)
    
    # 新Part1: 引言 + CLI接口规范
    new_part1 = part1_lines[0:1036]
    
    # 查找Python API开始位置（第3章）
    python_api_start = None
    for i, line in enumerate(part1_lines):
        if line.strip().startswith('## 3. Python API'):
            python_api_start = i
            break
    
    print(f"Python API起始行: {python_api_start}")
    
    # 查找异常体系开始位置（第4章）
    exception_start = None
    for i, line in enumerate(part1_lines):
        if line.strip().startswith('## 4. 异常体系'):
            exception_start = i
            break
    
    print(f"异常体系起始行: {exception_start}")
    
    # 新Part2: Python API核心（网格解析+求解器+配置管理）
    if python_api_start and exception_start:
        new_part2_header = [
            "# AutoFlowCFD 接口文档 - Part2\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，Python API核心接口（网格解析、求解器、配置管理）|\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        new_part2_content = part1_lines[python_api_start:exception_start]
        new_part2 = new_part2_header + new_part2_content
        
        # 添加跨文档引用
        new_part2_footer = [
            "\n---\n",
            "\n",
            "**文档结束 - Part2**\n",
            "\n",
            "*继续查看 [接口文档 - Part3](./2-4_接口文档-Part3.md) 了解后处理API、边界条件API和异常体系。*\n",
            "\n",
            "*返回 [接口文档 - Part1](./2-4_接口文档-Part1.md) 查看CLI接口规范。*\n"
        ]
        new_part2.extend(new_part2_footer)
    
    # 新Part3: Python API后处理+边界条件+异常体系
    if exception_start:
        new_part3_header = [
            "# AutoFlowCFD 接口文档 - Part3\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，Python API后处理、边界条件和异常体系|\n",
            "\n",
            "---\n",
            "\n",
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档的第三部分，涵盖后处理模块API、边界条件管理API和异常体系。\n",
            "\n",
            "**相关文档**:\n",
            "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
            "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        # 从原Part1提取异常体系
        exception_content = part1_lines[exception_start:]
        
        # 从原Part2提取后处理和边界条件API
        postprocess_start = None
        boundary_start = None
        
        for i, line in enumerate(part2_lines):
            if '## 2. 后处理模块 API' in line:
                postprocess_start = i
            elif '## 3. 边界条件管理 API' in line:
                boundary_start = i
        
        print(f"后处理API起始行(Part2): {postprocess_start}")
        print(f"边界条件API起始行(Part2): {boundary_start}")
        
        if postprocess_start and boundary_start:
            postprocess_content = part2_lines[postprocess_start:boundary_start]
            boundary_content = part2_lines[boundary_start:]
            
            # 合并内容
            new_part3_content = postprocess_content + ["\n---\n\n"] + boundary_content + ["\n---\n\n"] + exception_content
            
            new_part3 = new_part3_header + new_part3_content
            
            # 添加跨文档引用
            new_part3_footer = [
                "\n---\n",
                "\n",
                "**文档结束 - Part3**\n",
                "\n",
                "*继续查看 [接口文档 - Part4](./2-4_接口文档-Part4.md) 了解数据转换接口、插件扩展和错误码索引。*\n",
                "\n",
                "*返回 [接口文档 - Part2](./2-4_接口文档-Part2.md) 查看Python API核心接口。*\n"
            ]
            new_part3.extend(new_part3_footer)
    
    # 新Part4: 数据转换+插件扩展+错误码
    data_transform_start = None
    plugin_start = None
    error_code_start = None
    
    for i, line in enumerate(part2_lines):
        if '## 4. 数据转换与序列化接口' in line:
            data_transform_start = i
        elif '## 5. 插件扩展接口' in line:
            plugin_start = i
        elif '## 6. 错误码详细索引' in line:
            error_code_start = i
    
    print(f"数据转换接口起始行(Part2): {data_transform_start}")
    print(f"插件扩展接口起始行(Part2): {plugin_start}")
    print(f"错误码索引起始行(Part2): {error_code_start}")
    
    if data_transform_start and plugin_start and error_code_start:
        new_part4_header = [
            "# AutoFlowCFD 接口文档 - Part4\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，数据转换、插件扩展和错误码索引|\n",
            "\n",
            "---\n",
            "\n",
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档的第四部分，涵盖数据转换接口、插件扩展机制和错误码详细索引。\n",
            "\n",
            "**相关文档**:\n",
            "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
            "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口\n",
            "- [接口文档 - Part3](./2-4_接口文档-Part3.md): 后处理API和异常体系\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        # 提取到版本管理之前的内容
        version_mgmt_start = None
        for i, line in enumerate(part2_lines):
            if '## 7. 接口版本管理与兼容性' in line:
                version_mgmt_start = i
                break
        
        if version_mgmt_start:
            new_part4_content = part2_lines[data_transform_start:version_mgmt_start]
            new_part4 = new_part4_header + new_part4_content
            
            # 添加跨文档引用
            new_part4_footer = [
                "\n---\n",
                "\n",
                "**文档结束 - Part4**\n",
                "\n",
                "*继续查看 [接口文档 - Part5](./2-4_接口文档-Part5.md) 了解版本管理、兼容性保证和常见问题。*\n",
                "\n",
                "*返回 [接口文档 - Part3](./2-4_接口文档-Part3.md) 查看后处理API和异常体系。*\n"
            ]
            new_part4.extend(new_part4_footer)
    
    # 新Part5: 版本管理+FAQ+附录
    if version_mgmt_start:
        new_part5_header = [
            "# AutoFlowCFD 接口文档 - Part5\n",
            "\n",
            "## 文档版本控制\n",
            "\n",
            "|版本号|修订日期|修订人|修订说明|\n",
            "|---|---|---|---|\n",
            "|v0.2|2026-07-26|AutoFlowCFD团队|重构版本，版本管理、兼容性和常见问题|\n",
            "\n",
            "---\n",
            "\n",
            "## 1. 引言\n",
            "\n",
            "本文档为 **AutoFlowCFD** 接口设计规范文档的第五部分，涵盖接口版本管理、兼容性保证和常见问题解答。\n",
            "\n",
            "**相关文档**:\n",
            "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
            "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口\n",
            "- [接口文档 - Part3](./2-4_接口文档-Part3.md): 后处理API和异常体系\n",
            "- [接口文档 - Part4](./2-4_接口文档-Part4.md): 数据转换和插件扩展\n",
            "\n",
            "---\n",
            "\n"
        ]
        
        appendix_start = None
        for i, line in enumerate(part2_lines):
            if '## 8. 附录' in line or '## 9. 参考文献' in line:
                appendix_start = i
                break
        
        if appendix_start:
            new_part5_content = part2_lines[version_mgmt_start:]
            new_part5 = new_part5_header + new_part5_content
            
            # 更新内部引用
            new_part5_final = []
            for line in new_part5:
                # 更新旧的文件引用
                line = line.replace('./3-1_接口文档-Part1.md', './2-4_接口文档-Part1.md')
                line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
                new_part5_final.append(line)
            
            # 添加跨文档引用
            new_part5_footer = [
                "\n---\n",
                "\n",
                "**文档结束 - Part5（最终部分）**\n",
                "\n",
                "*返回以下文档查看更多内容：*\n",
                "- [接口文档 - Part1](./2-4_接口文档-Part1.md): CLI接口规范\n",
                "- [接口文档 - Part2](./2-4_接口文档-Part2.md): Python API核心接口\n",
                "- [接口文档 - Part3](./2-4_接口文档-Part3.md): 后处理API和异常体系\n",
                "- [接口文档 - Part4](./2-4_接口文档-Part4.md): 数据转换和插件扩展\n"
            ]
            new_part5_final.extend(new_part5_footer)
            new_part5 = new_part5_final
    
    # 更新Part1的尾部引用
    new_part1_updated = []
    for line in new_part1:
        line = line.replace('./3-1_接口文档-Part2.md', './2-4_接口文档-Part2.md')
        new_part1_updated.append(line)
    
    # 写入新文件
    print("\n正在写入新文件...")
    write_file(base_dir / '2-4_接口文档-Part1-new.md', new_part1_updated)
    write_file(base_dir / '2-4_接口文档-Part2-new.md', new_part2)
    write_file(base_dir / '2-4_接口文档-Part3-new.md', new_part3)
    write_file(base_dir / '2-4_接口文档-Part4-new.md', new_part4)
    write_file(base_dir / '2-4_接口文档-Part5-new.md', new_part5)
    
    # 统计新文件行数
    print(f"\n新文件行数统计:")
    print(f"Part1-new: {len(new_part1_updated)} 行")
    print(f"Part2-new: {len(new_part2)} 行")
    print(f"Part3-new: {len(new_part3)} 行")
    print(f"Part4-new: {len(new_part4)} 行")
    print(f"Part5-new: {len(new_part5)} 行")
    
    print("\n重构完成！请检查新生成的文件。")

if __name__ == '__main__':
    main()
