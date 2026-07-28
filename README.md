# Solaris Language Translator（索拉里斯文字翻译器）

将《鸣潮》（Wuthering Waves）游戏中的索拉里斯拉海洛地区字体（64字符集包含大小写，0-9，%，‘.’）识别并翻译为中文的 Web 工具。
目前只字符库只有拉海洛地区，web端可根据游戏内截图，映射输入对应字符，素材越多越好，或者字体文件提取标准字符集
## 工作原理

### 整体流程

```
游戏截图 → 字符分割 → 特征提取 → 模板匹配 → 词典纠错 → Google翻译 → 中文结果
```

### 1. 字符分割（`split_into_chars`）

针对索拉里斯方形字体优化的垂直投影分割：

- **二值化融合**：自适应阈值 + Otsu 双路二值化取并集，适应不同亮度和对比度的截图
- **极性归一化**：自动检测文字颜色（白字黑底/黑字白底），统一为白字黑底
- **水平投影**：检测文字行区域
- **垂直投影**：每行内按投影间隙切分独立字符（间隙阈值=1px，极低阈值确保字符内部断笔不切开）
- **比例过滤**：剔除过窄（<4px）、过宽（>画面90%）、宽高比异常（<0.2或>3.0）的噪声

### 2. 特征匹配（`GameCharMatcher`）

6 维特征加权匹配，综合评分：

| 特征 | 权重 | 说明 |
|------|------|------|
| **HOG** | 25% | 梯度方向直方图，对形状相似字符（n/r、E/F）有强区分力 |
| **宽高比** | 20% | 索拉里斯文字宽度是核心区分特征（Y窄V宽） |
| **NCC** | 15% | 归一化互相关，双极性匹配（正色+反色） |
| **pHash** | 15% | 感知哈希，双极性匹配，对渲染差异有容错 |
| **Canny边缘** | 15% | 边缘重叠率，膨胀1px提高容错 |
| **投影轮廓** | 10% | 水平+垂直投影轮廓NCC |

### 3. 词典纠错（`spell_correct_word`）

- **候选列表评分**：利用匹配阶段的逐位置 Top-10 候选，与 2000+ 英文单词词典进行全词匹配
- 仅当所有字符都在对应位置的候选列表中才接受纠错
- 可选 PySpellChecker 兜底

### 4. 翻译

- 使用 Google Translate 免费 API（`translate.googleapis.com`），无需 API Key

## 项目结构

```
├── app.py                    # Flask Web 服务主入口
├── matcher.py                # 匹配引擎 + 词典纠错
├── char_mapping.py           # 字符映射表（64字符）
├── extract_templates.py      # 从游戏截图提取真实模板
├── generate_font_templates.py# 从 TTF 字体生成模板
├── test_image.py             # 命令行测试工具
├── char_db/                  # 字符模板库
│   ├── A1.png                # TTF 生成的大写 A 模板
│   ├── a.png                 # TTF 生成的小写 a 模板
│   ├── B1_game_1.png         # 游戏截图提取的 B 模板
│   └── ...                   # 总计 75+ 个模板
├── templates/index.html      # Web 前端界面
└── static/style.css          # 前端样式
```

## 安装

### 环境要求

- Python 3.8+
- Windows/Linux/macOS

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/ZaKo333/Solaris-language-Translator.git
cd Solaris-language-Translator

# 安装依赖
pip install flask opencv-python numpy pillow requests scikit-image imagehash
```

### 获取字体模板（二选一）

**方式 A：使用已有模板（推荐）**

`char_db/` 目录已包含 64 个 TTF 生成模板 + 游戏渲染模板，开箱即用。

**方式 B：从 TTF 字体文件生成**

如果你有索拉里斯字体文件（`.ttf`），可自行生成模板：

```bash
# 修改 generate_font_templates.py 中的 TTF_PATH 为你的字体路径
python generate_font_templates.py
```

## 使用方法

### 方式一：Web 界面

```bash
python app.py
```

打开浏览器访问 **http://127.0.0.1:5000**

**识别翻译**标签页：
1. 拖入游戏截图
2. 点击「识别并翻译」
3. 查看识别结果和中文翻译

**提取模板**标签页：
1. 拖入已知文字的游戏截图（如显示 "Banana" 的图）
2. 输入图片中的文字内容
3. 点击「提取模板」
4. 系统自动分割字符并保存为游戏渲染模板，下次识别时生效

### 方式二：命令行

```bash
# 测试单张截图
python test_image.py <截图路径>

# 示例
python test_image.py banana.png

# 提取游戏模板
python extract_templates.py <截图路径> <已知文字>

# 示例：从 banana.png 提取 "Banana" 的字符模板
python extract_templates.py banana.png Banana
```

## 如何积累游戏模板？

识别精度取决于模板与游戏实际渲染的匹配程度。TTF 字体渲染的模板（细、干净）与游戏实际渲染（粗、有发光效果）差异较大，建议从游戏截图中提取真实模板：

1. 找到游戏中**已知文字**的截图（如菜单标题、按钮文字）
2. 在 Web 界面的「提取模板」标签页上传
3. 输入对应的文字内容
4. 系统自动分割并保存模板
5. 重复此过程积累更多字符的多种渲染变体

每个字符的多个变体模板会共存，匹配器自动选择匹配度最高的。

