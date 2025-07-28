import re
import numpy as np

def extract_features(blocks, page_height):
    font_sizes = [block["font_size"] for block in blocks]
    max_font = max(font_sizes) if font_sizes else 1.0
    features = []
    raw_metadata = []
    for i, block in enumerate(blocks):
        text = block["text"]
        font_size = block["font_size"]
        x = block["x"]
        y = block["y"]
        height = block["height"]
        page = block["page"]
        rel_font = font_size / max_font
        is_bold = int(block.get("bold", False))
        is_italic = int(block.get("italic", False))
        cap_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1e-6)
        left_margin = x
        space_above = 0 if i == 0 else y - blocks[i-1]["y"] - blocks[i-1]["height"]
        space_below = 0 if i == len(blocks)-1 else blocks[i+1]["y"] - y - height
        pos_on_page = y / page_height
        regex_patterns = [
            r"^\(?[0-9]+\)",
            r"^[0-9]+\.",
            r"^[A-Z]\.",
            r"^[IVXLCDM]+\."
        ]
        regex_score = int(any(re.match(p, text.strip()) for p in regex_patterns))
        feature_vec = [
            rel_font,
            is_bold,
            is_italic,
            cap_ratio,
            left_margin,
            space_above,
            space_below,
            pos_on_page,
            regex_score
        ]
        features.append(feature_vec)
        raw_metadata.append(block)
    X = np.array(features)
    return X, raw_metadata
