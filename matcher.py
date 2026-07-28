"""
字符匹配引擎 v19
HOG + 投影轮廓 + 边缘 + 宽高比 + NCC + pHash + CLAHE + 词典纠错

投影轮廓（水平+垂直投影）能有效区分形状相似的字符：
- l: 垂直投影窄尖峰 ←→ p: 垂直投影宽（有圆碗）
- H/h: 水平投影不同 ←→ E/e: 水平投影不同
- 1: 垂直投影极窄 ←→ y: 垂直投影宽（V形+尾巴）
"""

import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from char_mapping import get_char_map

TARGET = (200, 200)
HOG_SIZE = (128, 128)
CHAR_TARGET_SIZE = 140
PROFILE_SIZE = 100  # 投影轮廓采样点数


class GameCharMatcher:
    def __init__(self, db_path="char_db"):
        self.db_path = db_path
        self.char_map = get_char_map()
        self.std_gray = {}
        self.std_hog = {}
        self.std_hashes = {}
        self.std_aspect = {}
        self.std_hproj = {}   # 水平投影轮廓
        self.std_vproj = {}   # 垂直投影轮廓
        self.initialized = False
        self._last_top3 = []
        self._last_candidates = []

    def isolate_and_normalize(self, img_array):
        """字符提取 + 尺寸归一化 + 极性统一 + CLAHE对比度归一化"""
        if img_array is None or img_array.size == 0:
            return None, 0.0

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # CLAHE自适应对比度归一化 — 对不同颜色/亮度的截图更鲁棒
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(blurred)
        # 用CLAHE后的图像做二值化，但保留原始模糊图做后续灰度匹配
        _, binary = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright = np.sum(binary > 0)
        dark = np.sum(binary == 0)
        if bright > dark:
            binary = cv2.bitwise_not(binary)
            blurred = 255 - blurred
            equalized = 255 - equalized

        coords = cv2.findNonZero(binary)
        if coords is None:
            canvas = np.full(TARGET, 128, dtype=np.uint8)
            return canvas, 1.0

        x, y, w, h = cv2.boundingRect(coords)
        aspect = w / max(h, 1)

        margin = 3
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(blurred.shape[1] - x, w + margin * 2)
        h = min(blurred.shape[0] - y, h + margin * 2)
        char_region = blurred[y:y+h, x:x+w]

        scale = CHAR_TARGET_SIZE / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        char_scaled = cv2.resize(char_region, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full(TARGET, 128, dtype=np.uint8)
        xo = (TARGET[0] - new_w) // 2
        yo = (TARGET[1] - new_h) // 2
        canvas[yo:yo+new_h, xo:xo+new_w] = char_scaled

        return canvas, aspect

    def extract_profiles(self, img_array):
        """
        提取水平和垂直投影轮廓。
        先把字符二值化，计算每行/每列的字符像素比例，
        缩放到统一长度。
        """
        binary = (img_array < 100).astype(np.uint8)  # 暗字亮底
        h_proj = np.sum(binary, axis=1)    # 每行字符像素数
        v_proj = np.sum(binary, axis=0)    # 每列字符像素数

        # 归一化到0-1
        h_proj = h_proj.astype(np.float32) / max(h_proj.max(), 1)
        v_proj = v_proj.astype(np.float32) / max(v_proj.max(), 1)

        # 缩放到统一长度
        h_resized = cv2.resize(h_proj.reshape(-1, 1), (1, PROFILE_SIZE),
                                interpolation=cv2.INTER_LINEAR).flatten()
        v_resized = cv2.resize(v_proj.reshape(-1, 1), (1, PROFILE_SIZE),
                                interpolation=cv2.INTER_LINEAR).flatten()

        return h_resized, v_resized

    def preprocess(self, img_array):
        result, _ = self.isolate_and_normalize(img_array)
        return result

    def extract_hog(self, img_array):
        small = cv2.resize(img_array, HOG_SIZE, interpolation=cv2.INTER_AREA)
        from skimage.feature import hog as skimage_hog
        return skimage_hog(
            small, orientations=9,
            pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            feature_vector=True
        )

    def load_database(self):
        print("[INFO] 加载字符数据库 (v20 方形字体优化：宽高比20%+悬空点合并)...")
        if not os.path.exists(self.db_path):
            print(f"[WARN] 目录不存在: {self.db_path}")
            return False

        files = [f for f in os.listdir(self.db_path)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not files:
            print("[WARN] 数据库为空")
            return False

        loaded = 0
        for fname in files:
            key = os.path.splitext(fname)[0]

            if key in self.char_map:
                label = self.char_map[key]
            elif '_font_' in key:
                char_part = key.split('_font_')[0]
                label = {'percent': '%', 'dot': '.'}.get(char_part, char_part)
                self.char_map[key] = label
            elif '_game_' in key:
                char_part = key.split('_game_')[0]
                # 通过映射表解析前缀（B1 → B, dot → ., percent → %）
                label = self.char_map.get(char_part, char_part)
                self.char_map[key] = label
            else:
                continue

            path = os.path.join(self.db_path, fname)
            try:
                pil_img = Image.open(path).convert('L')
            except:
                continue

            raw = np.array(pil_img, dtype=np.uint8)
            processed, aspect = self.isolate_and_normalize(raw)
            if processed is None:
                continue

            self.std_gray[key] = processed
            self.std_aspect[key] = aspect
            self.std_hashes[key] = imagehash.phash(Image.fromarray(processed))

            # 投影轮廓
            self.std_hproj[key], self.std_vproj[key] = self.extract_profiles(processed)

            try:
                self.std_hog[key] = self.extract_hog(processed)
            except:
                pass

            loaded += 1

        print(f"[INFO] 加载 {loaded} 个模板")
        self.initialized = loaded > 0
        return self.initialized

    def ncc(self, a, b):
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        ma, mb = np.mean(a), np.mean(b)
        ca, cb = a - ma, b - mb
        num = np.sum(ca * cb)
        den = np.sqrt(np.sum(ca**2) * np.sum(cb**2))
        return float(num / den) if den > 1e-10 else 0.0

    def hog_similarity(self, fd1, fd2):
        norm1 = np.linalg.norm(fd1)
        norm2 = np.linalg.norm(fd2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        return float(np.dot(fd1, fd2) / (norm1 * norm2))

    def profile_similarity(self, p_q, p_t):
        """投影轮廓相似度（NCC）"""
        return self.ncc(p_q, p_t)

    def aspect_similarity(self, a_q, a_t):
        if a_t <= 0 or a_q <= 0:
            return 0.0
        ratio = a_q / a_t
        if ratio > 1.0:
            ratio = 1.0 / ratio
        return ratio

    def edge_similarity(self, img_q, img_t):
        """Canny边缘相似度 — 基于边缘重叠率，对渲染差异更鲁棒"""
        # 确保输入是uint8
        q_uint8 = np.clip(img_q, 0, 255).astype(np.uint8)
        t_uint8 = np.clip(img_t, 0, 255).astype(np.uint8)
        q_edges = cv2.Canny(q_uint8, 30, 100)
        t_edges = cv2.Canny(t_uint8, 30, 100)

        # 膨胀边缘1像素提高容错度（游戏渲染和字体模板的边缘位置略有偏差）
        d_kernel = np.ones((3, 3), np.uint8)
        q_edges_d = cv2.dilate(q_edges, d_kernel, iterations=1)
        t_edges_d = cv2.dilate(t_edges, d_kernel, iterations=1)

        overlap = np.sum((q_edges_d > 0) & (t_edges_d > 0))
        total = np.sum((q_edges_d > 0) | (t_edges_d > 0))
        if total < 10:
            return 0.5  # 边缘太少，给中性分
        return overlap / total

    def match_single_char(self, char_img, threshold=0.40):
        """HOG(25%) + 投影轮廓(10%) + 边缘(15%) + 宽高比(20%) + NCC(15%) + pHash(15%)

        索拉里斯文字核心特征：
        - 全部字符严格等高 → 宽高比是区分关键（Y窄V宽、大写大/小写小）
        - 相似字符轮廓略不同 → 投影轮廓辅助区分
        - 游戏渲染 vs 字体渲染有差异 → 边缘匹配容错
        """
        if not self.initialized:
            return None, 0.0, {}

        query, query_aspect = self.isolate_and_normalize(char_img)
        if query is None:
            return None, 0.0, {}

        query_inv = 255 - query

        try:
            query_hog = self.extract_hog(query)
        except:
            query_hog = None

        q_hproj, q_vproj = self.extract_profiles(query)

        query_hash = imagehash.phash(Image.fromarray(query))
        query_hash_inv = imagehash.phash(Image.fromarray(query_inv))

        results = []
        for key, label in self.char_map.items():
            if key not in self.std_gray:
                continue

            # HOG
            hog_score = 0.0
            if query_hog is not None and key in self.std_hog:
                hog_score = self.hog_similarity(query_hog, self.std_hog[key])

            # 投影轮廓
            hp_score = self.profile_similarity(q_hproj, self.std_hproj[key])
            vp_score = self.profile_similarity(q_vproj, self.std_vproj[key])
            prof_score = hp_score * 0.5 + vp_score * 0.5

            # 宽高比
            asp_score = self.aspect_similarity(query_aspect, self.std_aspect.get(key, 1.0))

            # NCC（双极性）
            ncc_score = max(self.ncc(query, self.std_gray[key]),
                           self.ncc(query_inv, self.std_gray[key]))

            # pHash（双极性）
            pd = min(query_hash - self.std_hashes[key],
                    query_hash_inv - self.std_hashes[key])
            ph_score = max(0.0, 1.0 - pd / 40.0)

            # 边缘相似度（双极性）
            edge_score = max(self.edge_similarity(query, self.std_gray[key]),
                            self.edge_similarity(query_inv, self.std_gray[key]))

            # 加权组合（宽高比占比最高 — 索拉里斯文字宽度是核心区分特征）
            combined = (hog_score * 0.25 + prof_score * 0.10 + edge_score * 0.15 +
                       asp_score * 0.20 + ncc_score * 0.15 + ph_score * 0.15)

            results.append((label, combined, ncc_score, ph_score,
                           hog_score, hp_score, vp_score, asp_score, edge_score))

        if not results:
            return None, 0.0, {}

        results.sort(key=lambda x: x[1], reverse=True)
        self._last_top3 = [(r[0], r[1]) for r in results[:3]]
        # 存储全部候选（用于词级纠错）
        self._last_candidates = [(r[0], r[1]) for r in results[:10]]

        best = results[0]
        best_label, best_score = best[0], best[1]

        if len(results) > 1:
            # 检查 top 候选是否都是同一个字符（只是不同模板变体）
            # 如果是，就不惩罚（游戏模板会有多个同名变体）
            top_label = results[0][0]
            second_label = results[1][0]
            if top_label != second_label:
                margin = best_score - results[1][1]
                if margin < 0.03:
                    best_score *= 0.5

        details = {
            'hog': best[4], 'hproj': best[5], 'vproj': best[6],
            'aspect': best[7], 'ncc': best[2], 'phash': best[3],
            'edge': best[8]
        }

        if best_score < threshold:
            return None, best_score, details

        return best_label, best_score, details


# ===== 词典纠错 =====

# 常用英文单词（用于词级纠错）
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us", "hello", "world", "welcome",
    "yes", "no", "ok", "hi", "thanks", "please", "sorry", "help",
    "happy", "sad", "big", "small", "hot", "cold", "old", "new",
    "open", "close", "start", "stop", "go", "come", "find",
    "here", "there", "where", "what", "when", "why", "how",
    "left", "right", "up", "down", "in", "out",
    "yes", "no", "on", "off", "true", "false",
    "attack", "defense", "power", "speed", "hp", "mp", "exp",
    "level", "quest", "item", "weapon", "armor", "shield",
    "potion", "elixir", "scroll", "key", "door", "chest",
    "gold", "coin", "money", "shop", "buy", "sell",
    "player", "enemy", "boss", "ally", "friend", "foe",
    "fire", "water", "wind", "earth", "light", "dark",
    "magic", "sword", "bow", "staff", "axe", "spear",
    "enter", "exit", "save", "load", "menu", "option",
    "config", "setting", "volume", "brightness", "language",
    "english", "chinese", "japanese", "korean",
    "name", "title", "message", "dialog", "accept", "cancel",
    "confirm", "decline", "continue", "retry", "abort",
    "waiting", "loading", "saving", "processing", "complete",
    "success", "failed", "error", "warning", "info",
    "player1", "player2", "score", "rank", "stage",
}
# 额外短词（2-4字母常用词，补充上述集合可能遗漏的）
EXTRA_SHORT_WORDS = {
    "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "am", "is", "are", "was", "were", "been",
    "has", "had", "did", "does", "done",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "need", "dare",
    "let", "make", "set", "put", "get", "run", "sit", "hit", "win",
    "eat", "cut", "pay", "lay", "die", "lie",
    "own", "old", "new", "big", "far", "bad", "low", "top",
    "age", "air", "arm", "art", "bag", "bed", "bit", "box",
    "boy", "bus", "cap", "car", "cat", "cup", "day", "dog",
    "ear", "eat", "egg", "end", "eye", "fan", "fat", "few",
    "fit", "fly", "fun", "gas", "god", "gun", "guy", "hat",
    "her", "him", "hit", "hot", "ice", "ill", "joy", "key",
    "leg", "lip", "lot", "low", "map", "mix", "net", "nor",
    "nut", "oil", "old", "own", "pan", "pen", "pet", "pig",
    "pin", "pit", "pop", "pot", "raw", "red", "rim", "rod",
    "row", "rub", "rug", "rum", "run", "sad", "sat", "say",
    "sea", "set", "sir", "sit", "six", "ski", "sky", "son",
    "sum", "sun", "tap", "tax", "tea", "ten", "the", "tie",
    "tin", "tip", "toe", "ton", "too", "top", "toy", "try",
    "two", "use", "van", "war", "way", "wet", "who", "why",
    "win", "wit", "yet", "zoo",
    "able", "also", "area", "army", "away", "back", "ball", "band",
    "bank", "base", "bath", "bear", "beat", "been", "bell", "belt",
    "best", "bill", "bird", "blow", "blue", "boat", "body", "bomb",
    "bone", "book", "born", "boss", "both", "burn", "busy", "call",
    "calm", "came", "camp", "card", "care", "case", "cash", "cast",
    "cell", "chat", "chip", "city", "club", "clue", "coal", "coat",
    "code", "coin", "cold", "come", "cook", "cool", "copy", "core",
    "cost", "crew", "crop", "cure", "dare", "dark", "data", "date",
    "dead", "deal", "dear", "deep", "deer", "desk", "dial", "diet",
    "dirt", "dish", "disk", "dock", "does", "done", "door", "dose",
    "down", "drag", "draw", "drew", "drop", "drug", "drum", "dual",
    "dull", "dump", "dust", "duty", "each", "earn", "ease", "edge",
    "edit", "else", "emit", "envy", "even", "ever", "evil", "exam",
    "exec", "exit", "face", "fact", "fade", "fail", "fair", "fake",
    "fall", "fame", "fang", "fare", "farm", "fast", "fate", "fear",
    "feed", "feel", "fell", "felt", "file", "fill", "film", "find",
    "fine", "fire", "firm", "fish", "flag", "flat", "flee", "flex",
    "flow", "fold", "folk", "fond", "food", "fool", "foot", "ford",
    "fore", "fork", "form", "fort", "four", "free", "from", "fuel",
    "full", "fume", "fund", "fury", "fuse", "fuss", "gain", "gale",
    "game", "gang", "gaol", "gap", "garage", "gate", "gave", "gaze",
    "gear", "gene", "gift", "girl", "give", "glad", "glow", "glue",
    "goat", "goes", "gold", "golf", "gone", "good", "grab", "gray",
    "grew", "grid", "grim", "grin", "grip", "grow", "gulf", "gust",
    "guts", "hack", "hair", "half", "hall", "halt", "hand", "hang",
    "happ", "hard", "harm", "hate", "haul", "have", "haze", "hazy",
    "head", "heal", "heap", "hear", "heat", "heav", "heel", "held",
    "hell", "helm", "help", "herb", "here", "hero", "hide", "high",
    "hill", "hilt", "hint", "hire", "hold", "hole", "holy", "home",
    "hood", "hook", "hope", "horn", "host", "hour", "huge", "hull",
    "hung", "hunt", "hurt", "hype", "icon", "idea", "idle", "inch",
    "info", "into", "iron", "isle", "item", "jack", "jail", "jean",
    "joke", "join", "jury", "just", "keen", "keep", "kept", "kick",
    "kill", "kind", "king", "kiss", "kite", "knee", "knew", "knit",
    "knob", "knot", "know", "lace", "lack", "lake", "lamp", "land",
    "lane", "last", "late", "lawn", "lead", "leaf", "leak", "lean",
    "leap", "left", "lend", "lens", "less", "liar", "lick", "lied",
    "life", "lift", "like", "limb", "lime", "limp", "line", "link",
    "lion", "list", "live", "load", "loan", "lock", "logo", "long",
    "look", "loop", "lord", "lose", "loss", "lost", "lots", "loud",
    "love", "luck", "lung", "lure", "lurk", "lust", "made", "mail",
    "main", "make", "male", "mall", "malt", "mane", "many", "mare",
    "mark", "mars", "mask", "mass", "mate", "maze", "mean", "meat",
    "meet", "melt", "memo", "mend", "menu", "mere", "mesh", "mess",
    "mid", "might", "mile", "milk", "mill", "mind", "mine", "miss",
    "mist", "moan", "mode", "mold", "mood", "moon", "more", "moss",
    "most", "moth", "move", "much", "mule", "must", "nail", "name",
    "navy", "near", "neat", "neck", "need", "nest", "next", "nice",
    "nine", "node", "none", "norm", "nose", "note", "noun", "nude",
    "numb", "obey", "odds", "okay", "omen", "omit", "once", "only",
    "onto", "ooze", "open", "oral", "orca", "ours", "oust", "oval",
    "oven", "over", "pace", "pack", "page", "paid", "pail", "pain",
    "pair", "pale", "palm", "pane", "park", "part", "pass", "past",
    "path", "peak", "peel", "peer", "pest", "pick", "pile", "pine",
    "pink", "pipe", "plan", "play", "plea", "plot", "plow", "plug",
    "plus", "poem", "poet", "pole", "poll", "pond", "pool", "poor",
    "pope", "pork", "port", "pose", "post", "pour", "pray", "prey",
    "prop", "pull", "pulp", "pump", "pure", "push", "quit", "quiz",
    "race", "rack", "raft", "rage", "raid", "rail", "rain", "rake",
    "ramp", "rang", "rank", "rare", "rash", "rate", "read", "real",
    "reap", "rear", "reef", "reel", "rein", "rely", "rend", "rent",
    "rest", "rice", "rich", "ride", "rift", "ring", "riot", "rise",
    "risk", "road", "roam", "rock", "rode", "role", "roll", "roof",
    "room", "root", "rope", "rose", "ruin", "rule", "rump", "rung",
    "ruse", "rush", "rust", "safe", "saga", "sage", "said", "sail",
    "sake", "sale", "salt", "same", "sand", "sane", "save", "scam",
    "scan", "seal", "seat", "seed", "seek", "seem", "seen", "self",
    "sell", "send", "sent", "shed", "shin", "ship", "shoe", "shop",
    "shot", "show", "shut", "sick", "side", "sigh", "sign", "silk",
    "sill", "silt", "sing", "sink", "site", "size", "skit", "slap",
    "slat", "slay", "sled", "slim", "slip", "slit", "slob", "slot",
    "slow", "slug", "slum", "slur", "smog", "snap", "snip", "snob",
    "snow", "snub", "snug", "soak", "soap", "soar", "sock", "soda",
    "sofa", "soft", "soil", "sold", "sole", "some", "song", "soon",
    "sore", "sort", "soul", "sour", "span", "spar", "spec", "sped",
    "spin", "spit", "spot", "spur", "star", "stay", "stem", "step",
    "stew", "stir", "stop", "stub", "stud", "stun", "such", "suit",
    "sum", "sung", "sunk", "sure", "surf", "swan", "swap", "swim",
    "tail", "take", "tale", "talk", "tall", "tame", "tank", "tape",
    "task", "taxi", "team", "tear", "tell", "tend", "tent", "term",
    "test", "text", "than", "that", "them", "then", "they", "thin",
    "this", "tick", "tide", "tidy", "tied", "tier", "tile", "till",
    "tilt", "time", "tiny", "tire", "toad", "toes", "told", "toll",
    "tomb", "tone", "took", "tool", "tops", "tore", "torn", "toss",
    "tour", "town", "trap", "tray", "tree", "trim", "trip", "trod",
    "trot", "true", "tube", "tuck", "tuft", "tuna", "tune", "turn",
    "twin", "type", "ugly", "undo", "unit", "unto", "upon", "urge",
    "used", "user", "vain", "vale", "vane", "vary", "vast", "veil",
    "vein", "vent", "verb", "very", "vest", "veto", "vice", "view",
    "vine", "void", "volt", "vote", "wade", "wage", "wait", "wake",
    "walk", "wall", "wand", "want", "ward", "warm", "warn", "warp",
    "wart", "wash", "wave", "wavy", "waxy", "weak", "wean", "wear",
    "weed", "week", "weep", "weld", "well", "went", "were", "west",
    "what", "when", "whim", "whip", "whom", "wick", "wife", "wild",
    "will", "wilt", "wily", "wind", "wine", "wing", "wink", "wipe",
    "wire", "wise", "wish", "with", "woke", "wolf", "wood", "wool",
    "word", "wore", "work", "worm", "worn", "wove", "wrap", "wren",
    "wrist", "write", "wrote", "yard", "year", "yell", "your", "zeal",
    "zero", "zinc", "zone", "zoom",
}
# 合并到主集合
COMMON_WORDS.update(EXTRA_SHORT_WORDS)


def spell_correct_word(word, candidates_per_pos=None):
    """
    词典纠错 v2 — 基于候选列表评分的全词匹配。

    如果有逐位置候选列表（candidates_per_pos），
    就遍历所有同长度的 COMMON_WORDS，
    检查每个字符是否在对应位置的候选列表中，
    选平均置信度最高的词。

    否则用简单的 pyspellchecker 拼写检查。
    """
    if not word or len(word) < 2:
        return word

    # 先试试直接就是单词
    word_lower = word.lower()
    if word_lower in COMMON_WORDS:
        return word

    # === 如果有逐位置候选，用全词匹配 ===
    if candidates_per_pos and len(candidates_per_pos) == len(word):
        best_word = word
        best_score = 0.0

        for dict_word in COMMON_WORDS:
            if len(dict_word) != len(word):
                continue

            score = 0.0
            possible = True
            for pos, dict_char in enumerate(dict_word):
                # 检查这个字符是否在位置 pos 的候选列表中
                found = False
                for cand_char, cand_conf in candidates_per_pos[pos]:
                    if cand_char.lower() == dict_char.lower():
                        score += cand_conf
                        found = True
                        break
                if not found:
                    possible = False
                    break

            if possible:
                avg_score = score / len(word)
                if avg_score > best_score:
                    best_score = avg_score
                    best_word = dict_word

        if best_word != word:
            print(f"   📖 词级纠错: '{word}' → '{best_word}' (置信度{best_score:.1%})")
            return best_word
        return word

    # === 简单拼写检查（兜底）===
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
        if not word.isalpha():
            return word
        misspelled = spell.unknown([word_lower])
        if not misspelled:
            return word
        best = spell.correction(word_lower)
        if best and best != word_lower:
            print(f"   📖 简单纠错: '{word}' → '{best}'")
            return best
    except:
        pass

    return word


def spell_correct(text):
    """兼容旧接口"""
    return spell_correct_word(text)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from char_mapping import get_char_map

    m = GameCharMatcher()
    m.load_database()
    print(f"\n模板: {len(m.std_gray)} 个")

    cm = get_char_map()
    ok = 0
    files = [f for f in os.listdir(m.db_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    for fname in files:
        key = os.path.splitext(fname)[0]
        if key not in cm or key not in m.std_gray:
            continue
        path = os.path.join(m.db_path, fname)
        try:
            pil_img = Image.open(path).convert('L')
            raw = np.array(pil_img, dtype=np.uint8)
        except:
            continue
        r, c, _ = m.match_single_char(raw, threshold=0.0)
        if r == cm[key]:
            ok += 1
    print(f"自检: {ok}/{len(m.std_gray)} ✅")
