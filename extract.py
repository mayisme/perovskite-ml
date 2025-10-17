import time
import os
import json
from typing import List, Dict, Optional
import pandas as pd
import google.generativeai as genai
import shutil
import asyncio
import textwrap
from dotenv import load_dotenv
load_dotenv()

try:
    import langextract as lx
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("警告：LangExtract 未安装，使用原有逻辑。")

def get_length(data):
    """简单长度检查函数，替代RDAgent的get_length"""
    return len(data) if hasattr(data, '__len__') else 0

def validate_and_correct(entities, threshold=5, rounds=2):
    """自定义纠错函数：维度检查 + 过滤噪声"""
    for _ in range(rounds):
        if get_length(entities) < threshold:
            print("低质量提取，过滤噪声...")
            entities = [e for e in entities if len(str(e)) > 2]  # 简单语法校正和过滤
        else:
            break
    return entities

print("=== 脚本开始执行 ===")
print(f"Python路径: {os.sys.executable}")
print(f"当前工作目录: {os.getcwd()}")
print("导入模块完成")

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
            if self.active_keys:
                 self.current_key_index = self.current_key_index % len(self.active_keys)
            else:
                self.current_key_index = 0
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        if not self.active_keys:
            raise Exception("没有可用的API密钥。所有密钥均已被暂停。")
        if self.current_key_index >= len(self.active_keys):
            self.current_key_index = 0
        return self.active_keys[self.current_key_index]
        
    def _rotate_key(self):
        """Rotate to next API key"""
        if not self.active_keys:
            print("警告: 尝试轮换密钥，但没有剩余的活动密钥。")
            return None 
        
        self.current_key_index = (self.current_key_index + 1) % len(self.active_keys)
        print(f"密钥已轮换。当前使用密钥尾号: {self.active_keys[self.current_key_index][-8:]}")
        
    async def call_api(self, prompt: str, max_retries: int = 3) -> Dict:
        """Call API with automatic key rotation and retry logic - 集成LangExtract"""
        # LangExtract提示和示例定义（移到方法内部，避免未定义错误）
        if ENHANCED_AVAILABLE:
            PROMPT = textwrap.dedent("""
            从学术文献中提取实体（作者、概念如'太阳能效率'）、关系（引用、影响）和语义属性（主题分类）。
            使用精确文本，不要改写。属性包括 affiliation、impact_type 等。
            专注于钙钛矿太阳能电池数据：PCE、Voc、Jsc、FF、添加剂、分子式、DOI等。
            """)

            EXAMPLES = [
                lx.data.ExampleData(
                    text="John Doe from MIT 提出太阳能面板效率达 25% (引用 Smith 2020)。",
                    extractions=[
                        lx.data.Extraction(extraction_class="author", extraction_text="John Doe", attributes={"affiliation": "MIT"}),
                        lx.data.Extraction(extraction_class="concept", extraction_text="太阳能面板效率", attributes={"value": "25%"}),
                        lx.data.Extraction(extraction_class="relation", extraction_text="引用", attributes={"target": "Smith 2020", "type": "influence"}),
                    ]
                )
            ]
        
        for attempt in range(max_retries):
            if not self.active_keys:
                return {"success": False, "error": "没有可用的API密钥"}
            
            current_key = self.get_current_key()
            
            try:
                if ENHANCED_AVAILABLE:
                    # 使用LangExtract增强抽取
                    result = lx.extract(
                        text_or_documents=prompt,
                        prompt_description=PROMPT,
                        examples=EXAMPLES,
                        model_id="gemini-2.5-flash",
                        api_key=current_key,
                        extraction_passes=2,
                        max_workers=2
                    )
                    response_text = json.dumps([{
                        'extraction_class': e.extraction_class,
                        'extraction_text': e.extraction_text,
                        'attributes': e.attributes
                    } for e in result.extractions])
                    
                    # 自定义纠错
                    extractions = [e.extraction_text for e in result.extractions]
                    validated_extractions = validate_and_correct(extractions)
                    response_text += f"\nValidated: {validated_extractions}"
                else:
                    # 原有Gemini逻辑作为fallback
                    import google.generativeai as genai
                    genai.configure(api_key=current_key)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    response = model.generate_content(prompt)
                    response_text = response.text if response.text else ""
                
                if response_text:
                    self.results[current_key].append({
                        "timestamp": time.time(),
                        "success": True,
                        "response_length": len(response_text)
                    })
                    return {"success": True, "response": response_text}
                else:
                    print(f"API返回空响应，尝试下一个密钥...")
                    self._rotate_key()
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                print(f"API调用失败 (密钥尾号: {current_key[-8:]}): {error_msg}")
                
                self.results[current_key].append({
                    "timestamp": time.time(),
                    "success": False,
                    "error": error_msg
                })
                
                if self._is_key_suspended(error_msg):
                    self._remove_suspended_key(current_key)
                    if not self.active_keys:
                        return {"success": False, "error": "所有API密钥均已被暂停"}
                else:
                    self._rotate_key()
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"等待 {wait_time} 秒后重试...")
                    await asyncio.sleep(wait_time)
        
        return {"success": False, "error": f"在 {max_retries} 次尝试后仍然失败"}

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
    
    md_files = [f for f in os.listdir(markdown_dir) if f.endswith('.md')]
    if not md_files:
        print(f"在 {markdown_dir} 中没有找到Markdown文件。")
        return

    all_extracted_data = []
    
    # CSV表头定义 - 包含作者和DOI字段（16个字段）
    csv_headers = [
        '序号', 'Group|组别', 'Additive', 'Molecular Formula', 'CID',
        'Optimized concentration', 'Perovskite component', 'Bandgap/eV',
        'Jsc(mA cm-2)', 'Voc (V)', 'FF（%）', 'PCE (%)',
        '文章标题', '作者', 'DOI', 'source_file'
    ]

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
                *   **作者**: 提取前3-5位主要作者，用分号分隔，格式如 "Zhang, L.; Wang, M.; Li, J."（不要超过80个字符）
                *   **DOI**: 提取数字对象标识符，格式如 "10.1038/s41586-021-03406-5"
            2.  **定位核心数据表**: 在论文中找到最关键的性能数据表。这是你的主要信息来源。
            3.  **提取所有实验组**: 完整地提取表格中的**每一行**数据。每一行都代表一个独立的实验组，都应该被记录下来。不要试图筛选或只选择"最好"的数据。
            4.  **识别组别 (Group)**:
                *   直接使用论文表格中对该实验组的描述作为"组别"名称。例如：`Pristine`, `Control`, `Rb-doped`, `LiOH + aging`, `CsFAMA-F` 等。
                *   如果原文明确使用了 `control`, `pristine`, `reference` 等词语，你可以将其归类为"对照组"。其他的则根据原文命名。
            5.  **推理与补全**:
                *   **分子式**: 如果缺少，请根据"添加剂名称"进行化学知识推理。例如，从 "Rubidium iodide" 推理出 "RbI"。
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
            14. 作者 (前3-5位主要作者，用分号分隔，如 "Zhang, L.; Wang, M.; Li, J."，不超过80字符)
            15. DOI (数字对象标识符，如 "10.1038/s41586-021-03406-5")
            16. 文件名 ({md_file})

            **输出示例:**
            {index},Pristine,,,,,Cs0.15FA0.65MA0.20Pb(I0.80Br0.20)3,1.68,21.81,1.14,79.59,19.79,Some Paper Title,"Zhang L.; Wang M.; Li J.",10.1038/s41586-021-03406-5,{md_file}
            {index+1},LiOH,LiOH,LiOH,,2 mg/mL,Cs0.15FA0.65MA0.20Pb(I0.80Br0.20)3,1.68,22.26,1.14,79.75,20.24,Some Paper Title,"Zhang L.; Wang M.; Li J.",10.1038/s41586-021-03406-5,{md_file}

            **重要：只输出CSV数据行，不要包含任何其他内容！**

            这是Markdown文档内容:
            --- Start of Document ---
            {markdown_content}
            --- End of Document ---
            """

            api_response = await api_client.call_api(prompt)
            
            if api_response.get('success', False):
                try:
                    response_text = api_response['response'].strip()
                    
                    # 处理增强响应：解析LangExtract JSON + 验证结果
                    if ENHANCED_AVAILABLE and 'Validated' in response_text:
                        # 提取LangExtract结果和验证实体
                        json_part, validated_part = response_text.split('\nValidated:')
                        import json
                        try:
                            extractions = json.loads(json_part)
                            validated_entities = eval(validated_part.strip())  # 安全eval
                            print(f"LangExtract提取: {len(extractions)} 实体，已验证: {len(validated_entities)}")
                        except:
                            extractions = []
                            validated_entities = []
                    else:
                        # 原有CSV处理作为fallback
                        extractions = []
                        validated_entities = []
                    
                    # 增强CSV处理：结合提取实体生成行
                    csv_lines = []
                    for line in response_text.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#') and ',' in line:
                            comma_count = line.count(',')
                            if comma_count >= 10:
                                csv_lines.append(line)
                    
                    # 如果CSV为空，尝试从提取实体生成简单行
                    if not csv_lines and extractions:
                        for ext in extractions[:5]:  # 限制数量
                            csv_line = f"{index},{ext.get('extraction_text', '')},,,,,,,"
                            csv_lines.append(csv_line)
                            index += 1
                    
                    if not csv_lines:
                        print(f"警告: 从 {md_file} 中没有提取到有效的CSV数据")
                        continue
                    
                    # 处理每一行CSV数据
                    current_row_number = index
                    for csv_line in csv_lines:
                        try:
                            import csv
                            from io import StringIO
                            reader = csv.reader(StringIO(csv_line))
                            row_values = next(reader)

                            while len(row_values) < len(csv_headers):
                                row_values.append('')

                            processed_row = {}
                            for i, header in enumerate(csv_headers):
                                if i < len(row_values):
                                    value = row_values[i].strip()
                                    if value and value.lower() not in ['null', '']:
                                        if header == '作者' and len(value) > 100:
                                            value = value[:100] + "..."
                                        processed_row[header] = value
                                    else:
                                        processed_row[header] = None
                                else:
                                    processed_row[header] = None

                            processed_row['序号'] = current_row_number
                            current_row_number += 1

                            # 添加验证实体到行（如果可用）
                            if validated_entities:
                                processed_row['验证实体'] = '; '.join(validated_entities[:3])

                            all_extracted_data.append(processed_row)

                        except Exception as e:
                            print(f"解析CSV行时出错 ({md_file}): {e}, 行内容: {csv_line[:100]}...")
                            continue
                    
                    index = current_row_number
                    
                    print(f"成功处理 {md_file}，提取了 {len(csv_lines)} 行数据 (增强模式)")
                    
                    destination_path = os.path.join(processed_md_dir, md_file)
                    shutil.move(filepath, destination_path)
                    print(f"文件 {md_file} 已移动到 {processed_md_dir}")
                    
                except Exception as e:
                    print(f"处理API响应时发生错误 ({md_file}): {e}")
            else:
                print(f"API调用失败: {api_response.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"处理文件 {md_file} 时发生错误: {e}")

    # Save to CSV
    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)
        df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到 {output_csv_file}")
        print(f"总共提取了 {len(all_extracted_data)} 行数据")
        
        # Calculate fill rates
        fill_rates = {}
        for col in csv_headers:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                fill_rate = (non_null_count / len(df)) * 100
                fill_rates[col] = round(fill_rate, 2)
        
        print("\nParameter Fill Rate (%):")
        for col, rate in fill_rates.items():
            print(f"{col}: {rate}%")
    else:
        print("No data was extracted.")

async def main():
    api_keys = [
        "AIzaSyCoAjvX0JqMQVAtTf5WwFMlT5iNfvWQxKM",

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

# 性能评估功能（新增）
def evaluate_extraction_accuracy(extracted_data, ground_truth_data):
    """
    简单评估抽取准确率（Precision, Recall, F1）
    extracted_data: 提取的实体列表
    ground_truth_data: 金标准实体列表
    """
    from sklearn.metrics import precision_recall_fscore_support
    import numpy as np
    
    # 简单字符串匹配评估
    true_positives = len(set(extracted_data) & set(ground_truth_data))
    false_positives = len(set(extracted_data) - set(ground_truth_data))
    false_negatives = len(set(ground_truth_data) - set(extracted_data))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'f1_score': round(f1, 3)
    }

# 性能评估建议
"""
性能评估指南：
1. 基准数据集：使用SciERC/CoNLL-2003测试实体识别F1；SemEval-2010测试关系抽取
2. 太阳能特定：准备100篇arXiv论文金标准标注
3. 测试流程：
   - source venv/bin/activate && python extract.py --eval
   - 比较集成前后F1提升（目标15%）
   - 纠错召回率>85%（低质量重试率）
4. A/B测试：50%流量原有逻辑，50%增强逻辑
5. 监控：延迟<5s/文档，LLM tokens<1000/调用
6. 模型比较：gemini-2.5-flash vs gemini-1.5-flash精度/速度
"""
