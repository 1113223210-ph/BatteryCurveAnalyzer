# gradio_app_with_llm.py
import gradio as gr
import torch
from fastai.vision.all import *
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# ---------- 1. 加载电池曲线分类器 ----------
print("📸 加载分类器模型...")
classifier_path = Path('models/curve_classifier_updated.pkl')
# 如果文件不存在，尝试加载旧版本
if not classifier_path.exists():
    classifier_path = Path('models/curve_classifier.pkl')
learn = load_learner(classifier_path)
print(f"✅ 分类器加载成功，类别: {learn.dls.vocab}")

# ---------- 2. 加载 Qwen 语言模型 ----------
print("🧠 加载 Qwen 语言模型...")
# 使用你刚刚成功的路径（请确认路径是否正确）
qwen_path = os.path.expanduser('~/MinivLLM/models/Qwen3-0.6B')

# 加载分词器和模型（使用 local_files_only=True 强制本地加载）
tokenizer = AutoTokenizer.from_pretrained(
    qwen_path,
    trust_remote_code=True,
    local_files_only=True
)
model = AutoModelForCausalLM.from_pretrained(
    qwen_path,
    torch_dtype=torch.float16,      # 使用半精度节省显存
    device_map='auto',               # 自动分配到 GPU
    trust_remote_code=True,
    local_files_only=True
)
print("✅ Qwen 模型加载成功！")

# ---------- 3. 定义预测函数（分类 + 生成分析）----------
def analyze_curve(image):
    # 3.1 分类
    pred, pred_idx, probs = learn.predict(image)
    curve_type = str(pred)
    confidence = probs[pred_idx].item()

    # 格式化分类结果
    classification_result = {learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))}

    # 3.2 构造提示词
    prompt = f"""你是一位电池材料科学家。用户上传了一张电池曲线图，经模型识别为 **{curve_type}** 类型（置信度 {confidence:.2%}）。
请用专业但易懂的语言回答以下问题：
1. 这种曲线通常用来衡量电池的什么性能？
2. 实验组和对照组有什么区别？

回答："""

    # 3.3 调用 Qwen 生成回答
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # 3.4 返回结果（分类结果 + 生成分析）
    return classification_result, answer

# ---------- 4. 创建 Gradio 界面 ----------
with gr.Blocks(title="电池曲线智能分析系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔋 电池曲线智能分析系统
    上传一张锂电池曲线图（过电位、对称电池或全电池），模型会自动识别其类别，并生成专业的文字分析。
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="上传曲线图")
            submit_btn = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(num_top_classes=3, label="分类结果")
            text_output = gr.Textbox(label="智能分析", lines=10, placeholder="等待分析结果...")

    submit_btn.click(
        fn=analyze_curve,
        inputs=image_input,
        outputs=[label_output, text_output]
    )

    gr.Markdown("### 📌 示例图片")
    gr.Examples(
        examples=[
            ["data/overpotential/你的示例图片1.png"],
            ["data/symmetrical/你的示例图片2.png"],
            ["data/full_cell/你的示例图片3.png"],
        ],
        inputs=image_input,
        outputs=[label_output, text_output],
        fn=analyze_curve,
        cache_examples=False
    )

# ---------- 5. 启动应用 ----------
if __name__ == "__main__":
    demo.launch(share=False)
