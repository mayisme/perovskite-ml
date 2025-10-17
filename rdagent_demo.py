import os
from dotenv import load_dotenv
load_dotenv()

try:
    import rdagent
    from rdagent.components.benchmark.conf import BenchmarkSettings
    from rdagent.app import create_agent
    RDAgent_AVAILABLE = True
    print("RDAgent导入成功")
except ImportError as e:
    RDAgent_AVAILABLE = False
    print(f"RDAgent导入失败: {e}")
    print("请确保已安装rdagent: pip install git+https://github.com/microsoft/rd-agent.git")

def configure_litellm_for_gemini():
    """配置LiteLLM使用Gemini API"""
    os.environ['CHAT_MODEL'] = 'gemini-2.5-flash'
    os.environ['OPENAI_API_KEY'] = os.getenv('GEMINI_API_KEY')  # LiteLLM兼容Gemini密钥
    os.environ['OPENAI_API_BASE'] = 'https://generativelanguage.googleapis.com/v1beta'  # Gemini端点
    print("LiteLLM配置完成: 使用gemini-2.5-flash模型")

def demo_basic_agent():
    """RDAgent基本用法演示"""
    if not RDAgent_AVAILABLE:
        print("跳过RDAgent演示: 模块未安装")
        return
    
    # 配置基准设置
    config = BenchmarkSettings(
        bench_test_round=3,  # 3轮测试
        max_loop=5  # 最大循环次数
    )
    
    # 创建代理
    agent = create_agent(config)
    
    # 定义任务: 文献抽取示例
    task_description = """
    任务: 从钙钛矿太阳能文献中提取关键性能指标。
    输入: 学术Markdown文本
    输出: 结构化CSV: 组别,添加剂,分子式,PCE(%),Voc(V),Jsc(mA/cm2),FF(%),DOI
    使用gemini-2.5-flash模型，确保精确提取和纠错。
    """
    
    print("启动RDAgent任务...")
    try:
        # 运行代理
        result = agent.run(task_description)
        print(f"RDAgent任务完成: {result}")
        
        # 示例日志获取 (如果支持trace端点)
        print("示例日志: 代理执行了研究、编码、执行阶段")
        
    except Exception as e:
        print(f"RDAgent运行错误: {e}")
        print("可能需要Docker环境或完整配置")

def demo_debug_tools():
    """演示RDAgent调试工具 (手动提取函数)"""
    # RDAgent debug_info_print实现 (从源代码提取)
    import sys
    import reprlib
    
    def get_original_code(func):
        """获取原始代码对象"""
        return func.__code__
    
    def debug_info_print(func):
        """RDAgent调试装饰器: 打印函数本地变量"""
        aRepr = reprlib.Repr()
        aRepr.maxother = 300
        
        def wrapper(*args, **kwargs):
            original_code = get_original_code(func)
            
            def local_trace(frame, event, arg):
                if event == "return" and frame.f_code == original_code:
                    print("\n" + "="*50)
                    print("RDAgent调试: 函数本地变量值:")
                    print("="*50)
                    for k, v in frame.f_locals.items():
                        printed = aRepr.repr(v)
                        print(f"{k}: {printed}")
                    print("="*50 + "\n")
                return local_trace
            
            sys.settrace(local_trace)
            try:
                return func(*args, **kwargs)
            finally:
                sys.settrace(None)
        return wrapper
    
    # RDAgent get_length实现
    def get_length(data):
        """RDAgent数据长度检查"""
        if hasattr(data, 'shape'):
            return data.shape[0]
        return len(data)
    
    # 演示使用
    @debug_info_print
    def sample_extraction(text):
        """示例抽取函数，演示调试"""
        entities = ['PCE: 25%', 'Voc: 1.14V', '作者: John Doe']
        print(f"提取实体: {entities}")
        return entities
    
    print("演示RDAgent调试工具...")
    sample_text = "示例太阳能文献"
    result = sample_extraction(sample_text)
    print(f"结果长度: {get_length(result)}")

if __name__ == "__main__":
    print("=== RDAgent用法演示 ===")
    print("1. LiteLLM配置 (Gemini API)")
    configure_litellm_for_gemini()
    
    print("\n2. 基本代理用法")
    demo_basic_agent()
    
    print("\n3. 调试工具演示 (手动提取)")
    demo_debug_tools()
    
    print("\n=== 演示完成 ===")
    print("注意: RDAgent完整功能需Docker环境和Azure配置")
    print("当前演示使用LiteLLM + Gemini 2.5-flash")
    print("推荐: 对于文献抽取，优先LangExtract + 自定义纠错")