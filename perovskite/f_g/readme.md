# 分子性质预测与模型分析平台

## 项目概述

本项目是一个基于机器学习的分子性质预测与模型分析平台，集成了多种功能包括：

- 分子性质预测（PCE性能预测）
- SHAP特征重要性分析
- 批量处理功能
- 实验设计方案生成
- 可视化分析

## 环境配置

### 1. 创建Python虚拟环境

```bash
python -m venv molecular_prediction
source molecular_prediction/bin/activate  # Linux/Mac
# 或
molecular_prediction\Scripts\activate      # Windows
```

### 2. 安装核心依赖

```bash
# 基础数据处理和机器学习库
pip install numpy pandas scikit-learn matplotlib seaborn plotly

# 化学信息学相关
pip install rdkit

# 机器学习框架
pip install xgboost catboost shap

# 深度学习框架（可选）
pip install torch torchvision torchaudio

# 其他工具
pip install joblib jupyter notebook

# Streamlit和相关依赖
pip install streamlit
pip install zhipuai  # 智谱AI API
```

### 3. 验证安装

```bash
python -c "import rdkit, streamlit, shap, xgboost, catboost; print('所有库安装成功')"
```

## 文件结构

```
项目根目录/
├── main_app.py              # 主Streamlit应用
├── shap_analyzer.py         # SHAP分析模块
├── catboost_model.py        # CatBoost模型训练和预测
├── data/
│   ├── 3.csv               # 主要数据集
│   ├── bit_to_functional_group.csv  # 官能团映射文件
│   └── output_20250826_130253/     # 模型输出目录
├── requirements.txt        # 依赖列表
└── README.md              # 本文档
```

## 数据准备

### 1. 数据集文件

确保以下文件存在于指定位置：

- `3.csv` - 主要分子数据集
- `bit_to_functional_group.csv` - 官能团映射文件
- `output_20250826_130253/trained_model_xgboost.pkl` - 预训练模型

### 2. 数据格式

数据集应包含以下列：

- `JV_default_PCE` - 目标变量（光电转换效率）
- `fp_*` - 指纹特征列（2048位）
- 其他分子描述符特征

## 使用教程

### 1. 启动主应用

```bash
streamlit run main_app.py
```

应用将在 http://localhost:8501 启动

### 2. 单次预测

1. 在侧边栏设置PCE阈值并选择基准行
2. 在"单次预测"标签页输入SMILES字符串
3. 点击"开始预测"查看结果

### 3. 批量预测

1. 准备SMILES文件（每行一个任务，多个SMILES用分号分隔）
2. 在"批量预测"标签页上传文件
3. 点击"开始批量预测"
4. 查看结果并导出数据

### 4. SHAP分析

1. 在侧边栏点击"生成SHAP分析图"
2. 等待分析完成后查看各类型图表
3. 可下载生成的图表

### 5. 命令行使用

```bash
# 单次预测
python shap_analyzer.py single "CCO" 0

# 批量预测
python shap_analyzer.py batch "CCO;CCN;CCC" 0

# 文件批量预测
python shap_analyzer.py file smiles.txt 0
```

### 6. CatBoost模型使用

```bash
python catboost_model.py
```

## 配置说明

### 1. API密钥配置

在 `catboost_model.py` 中配置智谱AI API密钥：

```python
client = ZhipuAI(api_key="your_api_key_here")
```

### 2. 文件路径配置

根据实际文件位置修改以下路径：

- 数据集路径
- 模型文件路径
- 输出目录路径

## 常见问题解决

### 1. RDKit安装问题

如果RDKit安装失败，可以尝试：

```bash
pip install rdkit
# 或者
pip install rdkit-pypi
```

### 2. SHAP依赖问题

确保安装正确版本的SHAP：

```bash
pip install shap==0.44.0
```

### 3. 内存不足问题

对于大型数据集，建议增加内存或使用数据采样：

```python
# 在代码中添加内存优化选项
import joblib
joblib.Memory(location='./cachedir', verbose=0)
```

### 4. 图形显示问题

如果图形无法显示，安装以下依赖：

```bash
pip install Pillow
```

## 功能特性

### 核心功能

- ✅ 分子指纹生成与处理
- ✅ 机器学习模型预测
- ✅ SHAP特征重要性分析
- ✅ 批量处理支持
- ✅ 数据可视化
- ✅ 结果导出

### 高级功能

- ✅ 官能团贡献分析
- ✅ 多SMILES合并处理
- ✅ 实验方案自动生成
- ✅ 性能指标计算
- ✅ 基准行选择优化

## 性能优化建议

1. 启用缓存：Streamlit的 `@st.cache_resource` 和 `@st.cache_data` 装饰器
2. 数据采样：对于大型数据集使用采样分析
3. 并行处理：利用多线程进行批量处理
4. 内存管理：及时清理不需要的数据

## 开发说明

### 代码结构

- 模块化设计：各功能模块分离
- 错误处理：完善的异常处理机制
- 类型提示：使用Python类型提示
- 文档字符串：详细的函数文档

### 扩展性

- 易于添加新的机器学习模型
- 支持多种分子描述符格式
- 可扩展的可视化选项
- 灵活的配置系统

## 技术支持

如遇到问题，请检查：

1. 所有依赖库版本是否兼容
2. 文件路径是否正确
3. 内存是否充足
4. API密钥是否有效

## 版本信息

- Python: 3.9+
- RDKit: 2023.03+
- Streamlit: 1.28+
- SHAP: 0.44+
- XGBoost: 1.7+
- CatBoost: 1.2+