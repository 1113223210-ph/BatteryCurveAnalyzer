# BatteryCurveAnalyzer
BatteryCurveAnalyzer 是一个结合**计算机视觉**与**大语言模型**的智能工具，能够自动识别锂电池测试曲线类型（过电位/对称电池/全电池），并生成专业的电化学分析报告，助力电池研究。
## ✨ 主要功能
- **图像分类**：基于 ResNet34 的迁移学习模型，准确识别三种电池曲线。
- **智能分析**：集成 Qwen3-0.6B 语言模型，根据分类结果生成现象解释、原因分析和测试建议。
- **Web 交互**：提供 Gradio 图形界面，上传图片即可获得分类结果+文字报告。
## ✨ 项目结构
```
BatteryCurveAI/
├── data/                     # 示例数据或数据存放说明（不包含实际图片）
├── models/                   # 训练好的分类器（.pkl）和下载说明
├── scripts/                  # Python 脚本
│   ├── train.py              # 训练分类器的脚本
│   ├── continue_train.py     # 微调脚本
│   ├── analyze.py            # 纯分类预测脚本
│   └── gradio_app.py         # Gradio Web 应用（集成 LLM）
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明文档
├── LICENSE                   # 许可证（如 MIT）
└── .gitignore                # 忽略临时文件、模型文件等
```
## 🚀 快速开始

### 环境配置
```bash
git clone https://github.com/1113223210-ph/BatteryCurveAI.git
cd BatteryCurveAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## 🌟 项目亮点
- **端到端流程**：从图像到文字分析，一站式解决。
- **专业提示词工程**：精心设计的 LLM 提示词，使回答更贴近电池领域。
- **可扩展性**：代码结构清晰，易于添加新的曲线类别或更换 LLM。
## 🔮 未来计划
- 增加更多曲线类型（如倍率性能、循环伏安等）。
- 引入 RAG 技术，让 LLM 基于真实论文知识回答。
- 部署到 Hugging Face Spaces 提供在线体验。
