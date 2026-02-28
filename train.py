from fastai.vision.all import *
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--model_path', type=str, default='../models')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    model_path.mkdir(exist_ok=True)

    # 加载数据
    dls = ImageDataLoaders.from_folder(
        data_path,
        valid_pct=0.2,
        seed=42,
        item_tfms=Resize(224),
        batch_tfms=aug_transforms(flip_vert=False, max_rotate=10, max_zoom=1.1),
        bs=args.batch_size
    )

    print(f"类别: {dls.vocab}")
    print(f"训练集: {len(dls.train_ds)} 张, 验证集: {len(dls.valid_ds)} 张")

    # 创建模型（使用预训练的 ResNet34）
    learn = vision_learner(dls, resnet34, metrics=[accuracy, Precision(average='macro'), Recall(average='macro'), F1Score(average='macro')])

    # 训练
    learn.fine_tune(args.epochs, args.lr)

    # 保存模型
    learn.export(model_path / 'curve_classifier.pkl')
    print(f"✅ 模型已保存至: {model_path / 'curve_classifier.pkl'}")

    # 绘制混淆矩阵
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(6,6))
    plt.savefig(model_path / 'confusion_matrix.png', dpi=120)
    print(f"✅ 混淆矩阵已保存: {model_path / 'confusion_matrix.png'}")

if __name__ == '__main__':
    main()
