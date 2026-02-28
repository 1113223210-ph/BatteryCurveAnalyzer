from fastai.vision.all import *
from pathlib import Path
import sys

# 加载模型（请确认路径正确）
model_path = Path(__file__).parent.parent / 'models' / 'curve_classifier.pkl'
learn = load_learner(model_path)

def classify_image(img_path):
    img = PILImage.create(img_path)
    pred, pred_idx, probs = learn.predict(img)
    print(f"\n📸 图片: {img_path}")
    print(f"🎯 预测类别: {pred}")
    print(f"📊 各类别置信度:")
    for i, cls in enumerate(learn.dls.vocab):
        print(f"   {cls}: {probs[i]:.4f}")
    return pred, probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze.py <图片路径>")
        sys.exit(1)
    classify_image(sys.argv[1])
