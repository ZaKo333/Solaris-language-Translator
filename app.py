"""
游戏文字翻译器 - Web服务
Flask + 字符模板匹配 + Google免费翻译
"""

import os
import uuid
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import requests
from matcher import spell_correct_word

app = Flask(__name__)

# ===== 配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['CHAR_DB_PATH'] = os.path.join(BASE_DIR, 'char_db')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['MATCH_THRESHOLD'] = 0.40

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHAR_DB_PATH'], exist_ok=True)

# ===== 初始化匹配器 =====
from matcher import GameCharMatcher
matcher = GameCharMatcher(db_path=app.config['CHAR_DB_PATH'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def imread_chinese(path, flags=cv2.IMREAD_GRAYSCALE):
    """支持中文路径的图片读取"""
    try:
        pil_img = Image.open(path)
        if flags == cv2.IMREAD_GRAYSCALE:
            pil_img = pil_img.convert('L')
        else:
            pil_img = pil_img.convert('RGB')
        return np.array(pil_img, dtype=np.uint8)
    except Exception:
        return None

def merge_nearby_boxes(boxes, line_height):
    """
    合并邻近的连通域 — 处理悬空点（如 i 的点、: 冒号）

    索拉里斯文字：
    - 全部字符严格等高、无笔画越出头
    - 相邻字符间有一定间隙（不会粘在一起）
    - 只有同一字符内部的断笔（如 i 的点）才需要合并

    所以合并条件严格限定在：
    1. 水平重叠 → 垂直接近 → 点与主体合并
    2. 垂直重叠 → 水平间隙<4px → 极近的断笔合并（绝不会合并不同字符）
    """
    if len(boxes) <= 1:
        return boxes

    boxes = list(boxes)
    merged = []

    while boxes:
        seed = boxes.pop(0)
        sx, sy, sw, sh = seed

        # 反复扫描，看其他框能否合并进来
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(boxes):
                bx, by, bw, bh = boxes[i]

                # 水平重叠量
                h_overlap = min(sx + sw, bx + bw) - max(sx, bx)
                # 垂直重叠量
                v_overlap = min(sy + sh, by + bh) - max(sy, by)

                should_merge = False

                # 情况1：水平重叠 + 垂直接近 → 悬空点（i的点、:的冒号）
                if h_overlap > 0:
                    v_gap = max(sy, by) - min(sy + sh, by + bh)
                    if v_gap > 0 and v_gap < line_height * 0.4:
                        should_merge = True

                # 情况2：垂直重叠 + 水平间隙极小 → 同一字符的断笔（如B的竖笔和碗）
                # 用较小部件的高度×0.14作为阈值：内部间隙约<高×0.14，字间距约>高×0.14
                if v_overlap > 0:
                    h_gap = max(sx, bx) - min(sx + sw, bx + bw)
                    h_ref = min(sh, bh)  # 用较小的部件高度做参考
                    gap_thresh = max(5, min(h_ref * 0.14, 12))
                    if h_gap > 0 and h_gap < gap_thresh:
                        should_merge = True

                # 情况3：一个部件宽度显著小于正常字符宽度（<行高×0.6），
                # 且合并后宽高比合理（<1.6）→ 是断开的同一字符（如非字结构）
                # 这能处理B的竖笔分离、R/T的左右分离等情况
                if not should_merge and v_overlap > 0:
                    # 至少一个部件宽度明显小于完整字符
                    narrow_part_w = min(sw, bw)
                    if narrow_part_w < line_height * 0.6:
                        h_gap = max(sx, bx) - min(sx + sw, bx + bw)
                        if h_gap > 0 and h_gap < 12:  # 间隙在合理范围内
                            combined_w = max(sx + sw, bx + bw) - min(sx, bx)
                            combined_h = max(sy + sh, by + bh) - min(sy, by)
                            if combined_h > 0 and combined_w / combined_h < 1.6:
                                should_merge = True

                if should_merge:
                    sx = min(sx, bx)
                    sy = min(sy, by)
                    sw = max(sx + sw, bx + bw) - sx
                    sh = max(sy + sh, by + bh) - sy
                    boxes.pop(i)
                    changed = True
                    break
                else:
                    i += 1

        merged.append((sx, sy, sw, sh))

    merged.sort(key=lambda b: b[0])
    return merged


def split_into_chars(image_path):
    """
    垂直投影法切割字符 — 针对索拉里斯方形字体优化

    索拉里斯字体特征：
    - 所有字符严格等高（无越出头笔画）
    - 字形略偏方形（宽高比接近1）
    - 相邻字符间有一定间隙，投影会降到0
    - 字符内部的断笔（如B的竖笔和碗）在投影上不会降到0
    - 部分有点（i, j, : 等），但点与主体水平重叠，投影不分割

    切割策略：
    1. 自适应阈值+Otsu融合二值化
    2. 水平投影检测行
    3. 每行内垂直投影检测字符边界（投影降到≈0处切开）
    4. 按比例过滤
    """
    img = imread_chinese(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h, w = img.shape
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # ===== 二值化 =====
    binary_adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4  # C=4，原参数
    )
    _, binary_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    fused = cv2.bitwise_or(binary_adaptive, binary_otsu)
    closed = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # ===== 极性统一：确保字符白色(255)，背景黑色(0) =====
    bright = np.sum(cleaned > 0)
    dark = np.sum(cleaned == 0)
    if bright > dark:
        cleaned = 255 - cleaned  # 反转

    # ===== 水平投影检测行 =====
    h_proj = np.sum(cleaned > 0, axis=1)
    h_thresh = max(5, w * 0.01)  # 降低到1%
    in_line = False
    line_ranges = []
    for y in range(h):
        if h_proj[y] > h_thresh and not in_line:
            line_start = y
            in_line = True
        elif h_proj[y] <= h_thresh and in_line:
            in_line = False
            if y - line_start > 5:
                line_ranges.append((line_start, y))
    if in_line:
        line_ranges.append((line_start, h - 1))
    line_ranges = [(y1, y2) for y1, y2 in line_ranges if y2 - y1 > 15]
    print(f"  [投影] 水平投影检测到 {len(line_ranges)} 行")

    if not line_ranges:
        # 兜底：如果行检测失败，用全图做垂直投影
        print(f"  [投影] ⚠ 行检测失败，尝试全图垂直投影")
        line_ranges = [(0, h)]
        return []

    # ===== 每行内垂直投影切割 =====
    chars = []
    for y1, y2 in line_ranges:
        line_h = y2 - y1
        line_img = cleaned[y1:y2, :]

        # 垂直投影
        v_proj = np.sum(line_img > 0, axis=0)
        v_max = int(np.max(v_proj))
        v_sum = int(np.sum(v_proj))
        # 极低阈值：字符内部断笔处投影会>0，字符间间隙=0
        v_thresh = 1  # 任何>0像素就算字符区域

        print(f"  [投影] 行{y1}-{y2}(高{line_h}) 垂直投影: max={v_max} sum={v_sum}")

        # 找出投影连续 > 阈值的区间 → 字符
        in_char = False
        char_starts, char_ends = [], []
        for x in range(w):
            if v_proj[x] > v_thresh and not in_char:
                char_starts.append(x)
                in_char = True
            elif v_proj[x] <= v_thresh and in_char:
                char_ends.append(x)
                in_char = False
        if in_char:
            char_ends.append(w - 1)

        print(f"  [投影] 字符区间数: {len(char_starts)}, 合并前区间: {list(zip(char_starts, char_ends))[:10]}")
        if not char_starts:
            continue

        # 合并极近的区间 —— 仅内部断笔才合并（间隙约一个字符宽度的6%，远小于字间距）
        if len(char_starts) > 1:
            ms, me = [char_starts[0]], []
            for i in range(1, len(char_starts)):
                gap = char_starts[i] - char_ends[i-1]
                avg_w = np.mean([char_ends[j] - char_starts[j] for j in range(i)])
                if gap < avg_w * 0.07 and avg_w > 0:
                    continue  # 内部断笔合并（%的两个圆圈等）
                else:
                    me.append(char_ends[i-1])
                    ms.append(char_starts[i])
            me.append(char_ends[-1])
            char_starts, char_ends = ms, me

        # 提取字符
        for xs, xe in zip(char_starts, char_ends):
            cw = xe - xs
            aspect = cw / max(line_h, 1)

            # 过滤检查
            skip_reason = None
            if cw < 4:
                skip_reason = f"太窄(cw={cw}<4)"
            elif cw > w * 0.9:
                skip_reason = f"太宽(cw>{w*0.9:.0f})"
            elif aspect > 3.0:
                skip_reason = f"比例({aspect:.2f}>3.0)"
            elif aspect < 0.2:
                skip_reason = f"比例({aspect:.2f}<0.2)"

            if skip_reason:
                print(f"  [投影]   ✗ 区间{xs}-{xe}(cw={cw}) {skip_reason}")
                continue

            print(f"  [投影]   ✓ 区间{xs}-{xe}(cw={cw} asp={aspect:.2f}) 通过")

            # 从原始模糊图上切
            patch = blurred[y1:y2, xs:xe]
            chars.append(pad_square(patch))

    return chars


def pad_square(char_img):
    """将字符图填充为正方形"""
    h, w = char_img.shape
    side = max(h, w) + 8
    square = np.zeros((side, side), dtype=np.uint8)
    xo = (side - w) // 2
    yo = (side - h) // 2
    square[yo:yo+h, xo:xo+w] = char_img
    return square


def translate_text(text, target="zh"):
    """Google Translate 免费接口（无需API Key）"""
    if not text or text.strip() == '?' * len(text):
        return text

    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        "client": "gtx", "sl": "en", "tl": target, "dt": "t", "q": text
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        translated = ""
        for part in result[0]:
            if part[0]:
                translated += part[0]
        return translated
    except Exception as e:
        print(f"[WARN] 翻译失败: {e}")
        return f"[翻译失败] {text}"


# ===== 路由 =====

@app.route('/')
def index():
    return render_template('index.html', stats={
        'char_count': len(matcher.std_gray),
        'ready': matcher.initialized
    })


@app.route('/api/translate', methods=['POST'])
def api_translate():
    """核心API：上传→识别→翻译"""
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片', 'success': False}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': '未选择文件', 'success': False}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式（支持PNG/JPG/BMP）', 'success': False}), 400
    if not matcher.initialized:
        return jsonify({'error': '字符数据库未初始化，请先放入标准图', 'success': False}), 500

    ext = file.filename.rsplit('.', 1)[1].lower()
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.{ext}")
    file.save(fpath)

    try:
        chars = split_into_chars(fpath)
        if not chars:
            return jsonify({'error': '未能识别出字符，请确认图片包含清晰文字', 'success': False}), 400
        if len(chars) > 200:
            return jsonify({'error': f'字符太多({len(chars)}个)，请裁剪范围', 'success': False}), 400

        results = []
        all_candidates = []
        for i, img in enumerate(chars):
            char, conf, details = matcher.match_single_char(img, threshold=app.config['MATCH_THRESHOLD'])
            # 收集逐位置候选（用于词级纠错）
            if hasattr(matcher, '_last_candidates') and matcher._last_candidates:
                all_candidates.append(matcher._last_candidates[:10])
            results.append({
                'index': i,
                'char': char or '?',
                'confidence': round(conf, 3),
                'details': {k: round(v, 3) for k, v in details.items()} if details else {}
            })

        raw = ''.join(r['char'] for r in results)
        # 带逐位置候选的词典纠错
        if len(all_candidates) == len(results):
            raw_corrected = spell_correct_word(raw, candidates_per_pos=all_candidates)
        else:
            raw_corrected = spell_correct_word(raw)
        if raw_corrected != raw:
            print(f"   📖 词典纠错: '{raw}' → '{raw_corrected}'")
            raw = raw_corrected
        translated = translate_text(raw)
        avg_conf = np.mean([r['confidence'] for r in results])
        uncertain = sum(1 for r in results if r['char'] == '?')

        return jsonify({
            'success': True,
            'raw_text': raw,
            'translated': translated,
            'char_count': len(results),
            'uncertain_count': uncertain,
            'avg_confidence': round(float(avg_conf), 3),
            'details': results
        })

    except Exception as e:
        return jsonify({'error': f'处理出错: {str(e)}', 'success': False}), 500
    finally:
        try:
            os.remove(fpath)
        except:
            pass


@app.route('/api/stats')
def api_stats():
    return jsonify({
        'db_char_count': len(matcher.std_gray),
        'initialized': matcher.initialized,
        'mapped_chars': len(set(matcher.char_map.values()))
    })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    global matcher
    matcher = GameCharMatcher(db_path=app.config['CHAR_DB_PATH'])
    ok = matcher.load_database()
    return jsonify({'success': ok, 'char_count': len(matcher.std_gray)})


@app.route('/api/extract_templates', methods=['POST'])
def api_extract_templates():
    """
    从已知文字的游戏截图中提取真实字符模板。
    上传图片 + 提供已知文字 → 自动分割、保存、更新映射、重载匹配器。
    """
    if 'image' not in request.files:
        return jsonify({'error': '未上传图片', 'success': False}), 400

    file = request.files['image']
    known_text = request.form.get('text', '').strip()

    if not known_text:
        return jsonify({'error': '请提供图片中的已知文字', 'success': False}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式', 'success': False}), 400

    # 保存上传
    ext = file.filename.rsplit('.', 1)[1].lower()
    fpath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.{ext}")
    file.save(fpath)

    try:
        # 分割字符
        chars = split_into_chars(fpath)
        if not chars:
            return jsonify({'error': '未能分割出字符，请确认图片包含清晰文字', 'success': False}), 400

        if len(chars) != len(known_text):
            return jsonify({
                'error': f'分割出 {len(chars)} 个字符，但文字长度为 {len(known_text)}，不匹配',
                'char_count': len(chars),
                'text_length': len(known_text),
                'success': False
            }), 400

        # 确定文件名和保存路径
        def get_save_key(char):
            if char == '.':
                return "dot"
            elif char == '%':
                return "percent"
            elif char.isupper():
                return f"{char}1"
            else:
                return char

        def get_next_game_id(save_key):
            max_id = 0
            char_db = app.config['CHAR_DB_PATH']
            if os.path.exists(char_db):
                for fname in os.listdir(char_db):
                    if fname.startswith(f"{save_key}_game_"):
                        try:
                            idx = int(fname.split('_game_')[1].split('.')[0])
                            max_id = max(max_id, idx)
                        except:
                            pass
            return max_id + 1

        char_db = app.config['CHAR_DB_PATH']

        saved = []
        for i, (char_img, expected_char) in enumerate(zip(chars, known_text)):
            save_key = get_save_key(expected_char)
            game_id = get_next_game_id(save_key)
            game_key = f"{save_key}_game_{game_id}"
            save_path = os.path.join(char_db, f"{game_key}.png")
            Image.fromarray(char_img).save(save_path)

            saved.append({
                'index': i,
                'char': expected_char,
                'game_key': game_key,
                'size': f"{char_img.shape[1]}x{char_img.shape[0]}"
            })

        # 重载匹配器（_game_ 命名的模板会被 load_database 自动识别）
        global matcher
        matcher = GameCharMatcher(db_path=char_db)
        matcher.load_database()

        return jsonify({
            'success': True,
            'saved': saved,
            'total': len(saved),
            'db_char_count': len(matcher.std_gray)
        })

    except Exception as e:
        return jsonify({'error': f'处理出错: {str(e)}', 'success': False}), 500
    finally:
        try:
            os.remove(fpath)
        except:
            pass


@app.route('/api/delete_ttf_templates', methods=['POST'])
def api_delete_ttf_templates():
    """删除所有TTF生成的模板（_font_ 和非 _game_ 的模板），只保留游戏截图提取的模板"""
    char_db = app.config['CHAR_DB_PATH']
    removed = 0
    mapping_path = os.path.join(BASE_DIR, 'char_mapping.py')

    for fname in os.listdir(char_db):
        if '_game_' not in fname:  # 不是游戏模板 → TTF生成的
            fpath = os.path.join(char_db, fname)
            try:
                os.remove(fpath)
                removed += 1
            except:
                pass

    # 重写 char_mapping.py，只保留 _game_ 条目
    with open(mapping_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_map = False
    for line in lines:
        if 'CHAR_MAP = {' in line:
            in_map = True
            new_lines.append(line)
        elif in_map and line.strip() == '}':
            new_lines.append(line)
            in_map = False
        elif in_map:
            # 只保留 _game_ 条目或非模板条目（函数定义等）
            if '_game_' in line or '#' in line or line.strip() == '':
                new_lines.append(line)
            # 跳过TTF条目（非 _game_ 的映射）
        else:
            new_lines.append(line)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    # 重载
    global matcher
    matcher = GameCharMatcher(db_path=char_db)
    matcher.load_database()

    return jsonify({
        'success': True,
        'removed': removed,
        'remaining': len(matcher.std_gray)
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  游戏文字翻译器 - Web服务")
    print("=" * 50)
    matcher.load_database()
    print(f"  数据库: {len(matcher.std_gray)} 个字符模板")
    print(f"  状态: {'就绪' if matcher.initialized else '未初始化'}")
    if not matcher.initialized:
        print("\n  ⚠ 请将字符图片放入 char_db/ 并编辑 char_mapping.py")
    print("\n  启动: http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
