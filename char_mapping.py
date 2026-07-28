"""
字符映射表
游戏内的自造字符 → 标准英文字母/数字/标点
"""

CHAR_MAP = {
    # ===== 大写字母 A-Z (TTF生成) =====
    'A1': 'A',
    'B1': 'B',
    'C1': 'C',
    'D1': 'D',
    'E1': 'E',
    'F1': 'F',
    'G1': 'G',
    'H1': 'H',
    'I1': 'I',
    'J1': 'J',
    'K1': 'K',
    'L1': 'L',
    'M1': 'M',
    'N1': 'N',
    'O1': 'O',
    'P1': 'P',
    'Q1': 'Q',
    'R1': 'R',
    'S1': 'S',
    'T1': 'T',
    'U1': 'U',
    'V1': 'V',
    'W1': 'W',
    'X1': 'X',
    'Y1': 'Y',
    'Z1': 'Z',

    # ===== 小写字母 a-z (TTF生成) =====
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'g',
    'h': 'h',
    'i': 'i',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'p': 'p',
    'q': 'q',
    'r': 'r',
    's': 's',
    't': 't',
    'u': 'u',
    'v': 'v',
    'w': 'w',
    'x': 'x',
    'y': 'y',
    'z': 'z',

    # ===== 数字 0-9 (TTF生成) =====
    '0': '0',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',

    # ===== 标点符号 (TTF生成) =====
    'dot': '.',
    'percent': '%',
}



def get_char_map():
    return CHAR_MAP


def get_reverse_map():
    """反向映射：英文字符 → 所有对应的图片文件名列表"""
    rev = {}
    for fname, char in CHAR_MAP.items():
        if char not in rev:
            rev[char] = []
        rev[char].append(fname)
    return rev


if __name__ == "__main__":
    print(f"当前映射了 {len(CHAR_MAP)} 个图片 → {len(set(CHAR_MAP.values()))} 个不同英文字符")
    print(f"英文字符列表: {''.join(sorted(set(CHAR_MAP.values())))}")
