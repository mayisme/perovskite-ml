import time
import os
import json
from typing import List, Dict, Optional
import pandas as pd
import google.generativeai as genai
import shutil # Added for moving files
import asyncio


class GeminiMultiAPI:
    def __init__(self, api_keys: List[str]):
        """Initialize with multiple API keys"""
        self.api_keys = api_keys.copy()
        self.active_keys = api_keys.copy()
        self.current_key_index = 0
        self.results: Dict[str, list] = {key: [] for key in api_keys}
        
    def _is_key_suspended(self, error_msg: str) -> bool:
        """Check if error indicates API key is suspended"""
        return "CONSUMER_SUSPENDED" in error_msg or "Permission denied" in error_msg or "API key not valid" in error_msg
        
    def _remove_suspended_key(self, key: str):
        """Remove a suspended API key from active rotation"""
        if key in self.active_keys:
            self.active_keys.remove(key)
            print(f"警告: API密钥尾号 {key[-8:]} 已被暂停，并已从轮换中移除。")
            # Reset index to avoid IndexError if current_key_index is now out of bounds
            if self.active_keys:
                 self.current_key_index = self.current_key_index % len(self.active_keys)
            else:
                self.current_key_index = 0
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        if not self.active_keys:
            raise Exception("没有可用的API密钥。所有密钥均已被暂停。")
        # Ensure current_key_index is valid
        if self.current_key_index >= len(self.active_keys):
            self.current_key_index = 0 # Reset if out of bounds (e.g., after key removal)
        return self.active_keys[self.current_key_index]
        
    def _rotate_key(self):
        """Rotate to next API key"""
        if not self.active_keys:
            # This case is handled by get_current_key, but good to have a check
            print("警告: 尝试轮换密钥，但没有剩余的活动密钥。")
            return None 
        
        self.current_key_index = (self.current_key_index + 1) % len(self.active_keys)
        print(f"密钥已轮换。当前使用密钥尾号: {self.active_keys[self.current_key_index][-8:]}")
        return self.active_keys[self.current_key_index]
        
    async def generate_content(self, prompt: str, model: str = 'gemini-2.5-flash', retries: int = 3) -> dict:
        """Generate content using a single API key (the current one)"""
        
        for attempt in range(retries):
            if not self.active_keys:
                return {
                    'api_key': 'N/A',
                    'success': False,
                    'error': "没有可用的API密钥。"
                }

            current_key = self.get_current_key()
            
            try:
                genai.configure(api_key=current_key)
                start_time = time.time()
                generative_model_instance = genai.GenerativeModel(model)
                generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
                
                response = await generative_model_instance.generate_content_async(
                    prompt,
                    generation_config=generation_config
                )
                
                elapsed_time = time.time() - start_time
                
                result = {
                    'api_key': current_key[-8:],
                    'success': True,
                    'elapsed_time': elapsed_time,
                    'response': response.text
                }
                
                # self.results[current_key].append(result) # Storing all results might consume a lot of memory
                return result
                    
            except Exception as e:
                error_msg = str(e)
                print(f"API调用错误 (密钥尾号 {current_key[-8:]}, 尝试 {attempt + 1}/{retries}): {error_msg}")
                
                if self._is_key_suspended(error_msg):
                    self._remove_suspended_key(current_key)
                    # Don't count this as a retry for the prompt, but rather a key failure.
                    # The next iteration of the loop (or a new call to generate_content) will use a new key.
                    # If we retry immediately, we might hit the same suspended key.
                    # Instead, we rotate and let the outer loop (processing files) handle the next attempt.
                    if not self.active_keys:
                         return {'api_key': current_key[-8:], 'success': False, 'error': "所有API密钥均已失效。" }
                    self._rotate_key() # Rotate to next key
                    # Continue to next attempt with a new key, or let the outer loop handle it
                    if attempt < retries -1: # if not the last attempt for this prompt
                        continue # try next attempt with potentially new key
                    else: # last attempt for this prompt failed
                         return {'api_key': current_key[-8:], 'success': False, 'error': f"API调用失败，所有重试均告失败: {error_msg}"}


                # For other errors, retry with the same key after a delay
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt) # Exponential backoff
                    self._rotate_key() # Rotate key before retrying for other errors too
                else: # Last retry failed
                    result = {
                        'api_key': current_key[-8:],
                        'success': False,
                        'error': error_msg
                    }
                    # self.results[current_key].append(result)
                    return result
        return {'api_key': 'N/A', 'success': False, 'error': '所有重试均告失败。'}


async def process_markdown_files(api_keys: List[str], markdown_dir: str, output_csv_file: str, processed_md_dir: str):
    """
    Process Markdown files using API to extract features for solar cell data.

    Args:
        api_keys: List of API keys
        markdown_dir: Directory containing Markdown files to process
        output_csv_file: Path to save the output CSV file
        processed_md_dir: Directory to move processed Markdown files
    """
    api_client = GeminiMultiAPI(api_keys)

    # Ensure processed_md_dir exists
    os.makedirs(processed_md_dir, exist_ok=True)

    # --- 调试代码开始 ---
    print(f"脚本启动...")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"正在检查Markdown目录: {os.path.abspath(markdown_dir)}")
    # --- 调试代码结束 ---

    print(f"DEBUG: All entries in '{markdown_dir}': {os.listdir(markdown_dir)}") # 调试代码
    md_files = [f for f in os.listdir(markdown_dir) if f.endswith('.md')]
    print(f"找到的Markdown文件: {md_files}") # 调试
    if not md_files:
        print(f"在 {markdown_dir} 中没有找到Markdown文件。")
        return

    all_extracted_data = []

    # CSV表头定义 - 每行代表一个实验组
    csv_headers = [
        '序号', 'Group|组别', 'Additive', 'Molecular Formula', 'CID',
        'Optimized concentration', 'Perovskite component', 'Bandgap/eV',
        'Jsc(mA cm-2)', 'Voc (V)', 'FF（%）', 'PCE (%)',
        '文章标题', '作者', 'DOI', 'source_file'
    ]

    # CSV示例格式（用于AI参考）- 多行格式
    csv_example = """1,对照组,,,,,CsFAMAif __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"主程序运行时出错: {e}"),1.62,22.1,1.12,75.2,18.5,Incorporation of rubidium cations into perovskite solar cells,markdown.md
2,优化组,Rubidium iodide (RbI),RbI,3423208,5%,RbCsFAMA,1.63,23.5,1.186,77.0,20.6,Incorporation of rubidium cations into perovskite solar cells,markdown.md"""

    for index, md_file in enumerate(sorted(md_files), 1):
        print(f"Processing {md_file}...")
        filepath = os.path.join(markdown_dir, md_file)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            prompt = f"""
            你是一位顶级的钙钛矿领域科研专家，拥有强大的信息提取和逻辑推理能力。
            你的核心任务是：深入分析下面的学术论文，并从文中包含关键性能对比的表格（通常包含PCE, Voc, Jsc, FF等参数）中，提取出所有实验组的数据，同时提取文章的基本信息（标题、作者、DOI）。

            **【数据提取指南】**

            1.  **提取文章基本信息**:
                *   **标题**: 提取文章的完整标题（通常在文档开头的 # 标题）
                *   **作者**: 提取所有作者姓名，多个作者用分号分隔，格式如 "Zhang, L.; Wang, M.; Li, J."
                *   **DOI**: 提取数字对象标识符，格式如 "10.1038/s41586-021-03406-5"
            2.  **定位核心数据表**: 在论文中找到最关键的性能数据表。这是你的主要信息来源。
            3.  **提取所有实验组**: 完整地提取表格中的**每一行**数据。每一行都代表一个独立的实验组，都应该被记录下来。不要试图筛选或只选择“最好”的数据。
            4.  **识别组别 (Group)**:
                *   直接使用论文表格中对该实验组的描述作为“组别”名称。例如：`Pristine`, `Control`, `Rb-doped`, `LiOH + aging`, `CsFAMA-F` 等。
                *   如果原文明确使用了 `control`, `pristine`, `reference` 等词语，你可以将其归类为“对照组”。其他的则根据原文命名。
            5.  **推理与补全**:
                *   **分子式**: 如果缺少，请根据“添加剂名称”进行化学知识推理。例如，从 "Rubidium iodide" 推理出 "RbI"。
                *   **单位转换**: 确保所有数值都以纯数字（float 或 int）形式返回。例如，"1120 mV" 应提取为 `1.12` (V)；"81%" 应提取为 `81.0`。
                *   **缺失值**: 如果信息在文中确实无法找到，则返回 `null`。不要臆测。
            6.  **格式化输出**:
                *   严格按照CSV格式输出数据，每行代表一个实验组。
                *   **不要包含任何字段名、标题行或解释性文字**。只输出纯粹的CSV数据行。
                *   每行必须用逗号分隔16个字段。

            **字段顺序（16个字段）：**
            1. 序号 (从{index}开始，每行递增)
            2. 组别 (直接从论文表格中提取，如 `Pristine`, `Rb-doped`, `对照组` 等)
            3. 添加剂名称 (如 "Sodium fluoride", "Rubidium iodide"。对于对照组或不适用情况，留空)
            4. 分子式 (添加剂的分子式，如 "NaF", "RbI"。如果适用，可填写钙钛矿完整分子式)
            5. CID (化合物CID编号，如果适用)
            6. 优化浓度 (如 "5%", "0.1 mol%"。不适用则留空)
            7. 钙钛矿组分 (标准化学式，如 "CsFAMA", "Cs0.05FA0.54MA0.41Pb(I0.98Br0.02)3")
            8. 带隙 (eV，纯数字)
            9. 短路电流 (mA cm-2，纯数字)
            10. 开路电压 (V，纯数字，mV需转换)
            11. 填充因子 (%，纯数字)
            12. 电池效率 (%，纯数字)
            13. 文章标题 (完整标题)
            14. 作者 (作者姓名，多个作者用分号分隔，如 "Zhang, L.; Wang, M.; Li, J.")
            15. DOI (数字对象标识符，如 "10.1038/s41586-021-03406-5")
            16. 文件名 ({md_file})

            **输出示例:**
            {index},Pristine,,,,,Cs0.15FA0.65MA0.20Pb(I0.80Br0.20)3,1.68,21.81,1.14,79.59,19.79,Some Paper Title,Zhang L.; Wang M.; Li J.,10.1038/s41586-021-03406-5,{md_file}
            {index+1},LiOH,LiOH,LiOH,null,2 mg/mL,Cs0.15FA0.65MA0.20Pb(I0.80Br0.20)3,1.68,22.26,1.14,79.75,20.24,Some Paper Title,Zhang L.; Wang M.; Li J.,10.1038/s41586-021-03406-5,{md_file}
            {index+2},LiOH + aging,LiOH,LiOH,null,2 mg/mL,Cs0.15FA0.65MA0.20Pb(I0.80Br0.20)3,1.68,21.88,1.18,81.82,21.12,Some Paper Title,Zhang L.; Wang M.; Li J.,10.1038/s41586-021-03406-5,{md_file}

            **重要：只输出CSV数据行，不要包含任何其他内容！**

            **现在，请开始分析以下论文内容：**
            --- Start of Document ---
            {markdown_content}
            --- End of Document ---
            """
            
            api_response = await api_client.generate_content(prompt)
            
            if api_response.get('success', False):
                try:
                    response_text = api_response['response'].strip()

                    # 处理CSV格式的响应
                    csv_lines = []
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and ',' in line:
                            # 检查是否为有效的数据行（至少包含组别信息）
                            # The new prompt is designed to only return valid CSV lines.
                            # Any line with a comma is considered a potential data line.
                            csv_lines.append(line)

                    if not csv_lines:
                        print(f"警告: 从 {md_file} 中没有提取到有效的CSV数据")
                        continue

                    # 处理每一行CSV数据（现在每行代表一个实验组）
                    current_row_number = index
                    for csv_line in csv_lines:
                        try:
                            # 直接按逗号分割，不使用CSV解析器（避免引号问题）
                            row_values = csv_line.split(',')

                            # 确保行有足够的字段
                            while len(row_values) < len(csv_headers):
                                row_values.append('')

                            # 创建行数据字典
                            processed_row = {}
                            for i, header in enumerate(csv_headers):
                                if i < len(row_values):
                                    value = row_values[i].strip().strip('"')  # 移除引号
                                    if value and value.lower() != 'null':
                                        processed_row[header] = value
                                    else:
                                        processed_row[header] = None
                                else:
                                    processed_row[header] = None

                            # 更新序号为当前行号
                            processed_row['序号'] = current_row_number
                            current_row_number += 1

                            # 验证行是否包含实质性数据
                            key_fields = list(processed_row.keys())
                            if '序号' in key_fields:
                                key_fields.remove('序号')
                            if 'source_file' in key_fields:
                                key_fields.remove('source_file')
                            
                            if any(processed_row[key] is not None for key in key_fields):
                                all_extracted_data.append(processed_row)
                            else:
                                print(f"丢弃空的或无效的数据行: {csv_line}")


                        except Exception as e:
                            print(f"解析CSV行时出错 ({md_file}): {e}, 行内容: {csv_line}")
                            continue

                    # 更新全局索引
                    index = current_row_number

                    print(f"成功处理 {md_file}，提取了 {len(csv_lines)} 行数据")

                    # Move the processed file
                    destination_path = os.path.join(processed_md_dir, md_file)
                    shutil.move(filepath, destination_path)
                    print(f"文件 {md_file} 已移动到 {processed_md_dir}")

                except Exception as e:
                    print(f"处理API响应时发生错误 ({md_file}): {e}")
            else:
                print(f"API call failed for {md_file}: {api_response.get('error', 'Unknown error')}")
                        
        except Exception as e:
            print(f"Error reading or processing file {md_file}: {e}")
    
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data, columns=csv_headers)
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\nData saved to {output_csv_file}")
        
        print("\nExtraction Statistics:")
        fill_rates = (df.notna().sum() / len(df) * 100).round(2)
        print(f"Total articles processed: {len(df)}")
        print("\nParameter Fill Rate (%):")
        for col, rate in fill_rates.items():
            print(f"{col}: {rate}%")
    else:
        print("No data was extracted.")

async def main():
    api_keys = [
        "AIzaSyCoAjvX0JqMQVAtTf5WwFMlT5iNfvWQxKM",
        #"AIzaSyBIHQsTEWzQQ8UxU3IBf4WziaHsVUyq9gc",
        ]
    
    # Corrected directory paths
    unread_md_dir = "data"
    read_md_dir = "read_mds"
    output_csv = "extracted_solar_data.csv"
    print(f"脚本启动...")
    print(f"当前工作目录: {os.getcwd()}")


    await process_markdown_files(
        api_keys=api_keys,
        markdown_dir=unread_md_dir,
        output_csv_file=output_csv,
        processed_md_dir=read_md_dir
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"主程序运行时出错: {e}")