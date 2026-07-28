"""
特征提取器：用 ONNX Runtime + MobileNet 提取图像特征
不需要PyTorch，轻量级，无DLL冲突。

模型自动从 ONNX Model Zoo 下载（首次约14MB）。
"""

import os
import cv2
import numpy as np
from PIL import Image
import urllib.request

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2-7.onnx")
INPUT_SIZE = 224


def _download_model():
    """下载MobileNet v2 ONNX模型"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[INFO] 下载AI特征模型 (~14MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] 模型下载完成")
        return True
    except Exception as e:
        print(f"[WARN] 模型下载失败: {e}")
        return False


_session = None
_input_name = None
_output_name = None
_model_loaded = False


def init_model():
    """初始化ONNX模型"""
    global _session, _input_name, _output_name, _model_loaded

    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARN] onnxruntime未安装")
        return False

    if not os.path.exists(MODEL_PATH):
        if not _download_model():
            return False

    if not os.path.exists(MODEL_PATH):
        return False

    try:
        _session = ort.InferenceSession(MODEL_PATH)
        _input_name = _session.get_inputs()[0].name
        _output_name = _session.get_outputs()[0].name
        _model_loaded = True
        print("[INFO] AI特征模型就绪 ✓")
        return True
    except Exception as e:
        print(f"[WARN] 模型加载失败: {e}")
        return False


def extract_features(img_array):
    """
    提取图像特征向量（1024维）。
    返回L2归一化的特征向量，或None。
    """
    global _session, _input_name, _output_name, _model_loaded

    if not _model_loaded:
        if not init_model():
            return None

    if _session is None:
        return None

    try:
        # 缩放到模型输入尺寸
        if len(img_array.shape) == 2:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE),
                             interpolation=cv2.INTER_AREA)

        # 标准化 (像ImageNet训练时一样)
        blob = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        blob = (blob - mean) / std

        # NCHW格式
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        # 推理
        outputs = _session.run([_output_name], {_input_name: blob})
        features = outputs[0].flatten()

        # L2归一化
        norm = np.linalg.norm(features)
        return features / norm if norm > 1e-10 else features

    except Exception as e:
        print(f"[WARN] 特征提取失败: {e}")
        return None


def cosine_similarity(f1, f2):
    """余弦相似度（向量已归一化则直接点积）"""
    return float(np.dot(f1, f2))


if __name__ == "__main__":
    # 测试
    ok = init_model()
    print(f"模型就绪: {ok}")
    if ok:
        # 生成一个测试图
        test_img = np.full((200, 200), 128, dtype=np.uint8)
        feat = extract_features(test_img)
        print(f"特征维度: {len(feat) if feat is not None else '失败'}")
