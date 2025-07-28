# import pdfplumber
# from typing import List, Dict
# from collections import defaultdict
# import os
# import re
# import json
# import glob
# from features import extract_features
# from model import train_model

# def reconstruct_line_text(line_words: List[Dict], gap_multiplier: float = 0.3) -> str:
#     if not line_words:
#         return ""
#     line_words = sorted(line_words, key=lambda w: w["x0"])
#     widths = [w["x1"] - w["x0"] for w in line_words if len(w["text"]) > 0]
#     avg_char_width = sum(widths) / sum(len(w["text"]) for w in line_words if len(w["text"]) > 0)
#     space_threshold = avg_char_width * (1.0 + gap_multiplier)
#     pieces = [line_words[0]["text"]]
#     for i in range(1, len(line_words)):
#         prev = line_words[i - 1]
#         curr = line_words[i]
#         gap = curr["x0"] - prev["x1"]
#         if gap > space_threshold:
#             pieces.append(" ")
#         pieces.append(curr["text"])
#     return "".join(pieces)

# def fix_spaced_text(text: str) -> str:
#     if len(text) < 3 or '  ' in text or text.count(' ') < 2:
#         return text
#     tokens = text.strip().split()
#     if all(len(tok) == 1 for tok in tokens):
#         return ''.join(tokens)
#     return text

# def parse_pdf(file_path: str) -> List[Dict]:
#     parsed_blocks = []
#     with pdfplumber.open(file_path) as pdf:
#         for page_number, page in enumerate(pdf.pages, start=1):
#             words = page.extract_words(extra_attrs=["fontname", "size", "x0", "x1", "top", "bottom"])
#             lines = defaultdict(list)
#             for word in words:
#                 y_rounded = round(float(word["top"]) / 2) * 2
#                 lines[y_rounded].append(word)
#             for y_val, line_words in lines.items():
#                 if not line_words:
#                     continue
#                 line_words.sort(key=lambda w: w["x0"])
#                 line_text = " ".join(w["text"] for w in line_words)
#                 font_sizes = [float(w["size"]) for w in line_words]
#                 avg_font = sum(font_sizes) / len(font_sizes)
#                 bold_flags = ["Bold" in w["fontname"] for w in line_words]
#                 italic_flags = ["Italic" in w["fontname"] or "Oblique" in w["fontname"] for w in line_words]
#                 block = {
#                     "text": fix_spaced_text(line_text.strip()),
#                     "font_size": round(avg_font, 2),
#                     "x": float(line_words[0]["x0"]),
#                     "y": float(line_words[0]["top"]),
#                     "width": float(line_words[-1]["x1"]) - float(line_words[0]["x0"]),
#                     "height": float(line_words[0]["bottom"]) - float(line_words[0]["top"]),
#                     "bold": any(bold_flags),
#                     "italic": any(italic_flags),
#                     "fontname": line_words[0]["fontname"],
#                     "page": page_number
#                 }
#                 parsed_blocks.append(block)
#     return parsed_blocks

# all_blocks = []
# for file in glob.glob("labeled_pdfs/*.json"):
#     with open(file) as f:
#         all_blocks.extend(json.load(f))

# X, raw_metadata = extract_features(all_blocks, page_height=800)
# y = [b["label"] for b in raw_metadata]

# train_model(X, y, "trained_model.pkl")

import pdfplumber
from typing import List, Dict
from collections import defaultdict
import os
import re
from features import extract_features
from model import train_model, predict_headings
import json

def reconstruct_line_text(line_words: List[Dict], gap_multiplier: float = 0.3) -> str:
    """
    Reconstructs a line of text by inserting spaces based on horizontal gaps.
    Uses x0/x1 values of words to infer spacing.
    """
    if not line_words:
        return ""

    # Sort left to right
    line_words = sorted(line_words, key=lambda w: w["x0"])

    # Estimate average character width for spacing threshold
    widths = [w["x1"] - w["x0"] for w in line_words if len(w["text"]) > 0]
    avg_char_width = sum(widths) / sum(len(w["text"]) for w in line_words if len(w["text"]) > 0)
    space_threshold = avg_char_width * (1.0 + gap_multiplier)

    pieces = [line_words[0]["text"]]
    for i in range(1, len(line_words)):
        prev = line_words[i - 1]
        curr = line_words[i]

        gap = curr["x0"] - prev["x1"]
        if gap > space_threshold:
            pieces.append(" ")
        pieces.append(curr["text"])

    return "".join(pieces)

def fix_spaced_text(text: str) -> str:
    """
    Fix strings like 'O v e r v i e w' -> 'Overview' while preserving real multi-word text.
    Optimized for speed.
    """
    # Fast path: skip lines that already contain normal-length words
    if len(text) < 3 or '  ' in text or text.count(' ') < 2:
        return text  # Nothing suspicious

    # Heuristic: if all words are 1-char long (e.g., ['F','o','o']), merge
    tokens = text.strip().split()
    if all(len(tok) == 1 for tok in tokens):
        return ''.join(tokens)

    return text  # Likely valid

def parse_pdf(file_path: str) -> List[Dict]:
    """
    Parses a PDF and merges nearby words on the same line into full headings/text blocks.
    Returns a list of block dictionaries with text, font, position, and page metadata.
    """
    parsed_blocks = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["fontname", "size", "x0", "x1", "top", "bottom"])

            # Group words by line using y position buckets
            lines = defaultdict(list)
            for word in words:
                y_rounded = round(float(word["top"]) / 2) * 2  # Grouping by similar Y (~2px tolerance)
                lines[y_rounded].append(word)

            # print(lines.items())

            for y_val, line_words in lines.items():
                if not line_words:
                    continue
                # print(line_words + "\n")

                # Sort left to right
                line_words.sort(key=lambda w: w["x0"])
                line_text = " ".join(w["text"] for w in line_words)

                font_sizes = [float(w["size"]) for w in line_words]
                avg_font = sum(font_sizes) / len(font_sizes)

                bold_flags = ["Bold" in w["fontname"] for w in line_words]
                italic_flags = ["Italic" in w["fontname"] or "Oblique" in w["fontname"] for w in line_words]
                # print(line_text + "\n")

                block = {
                    # "text": fix_spaced_text(line_text.strip()),
                    "text": fix_spaced_text(line_text.strip()),
                    "font_size": round(avg_font, 2),
                    "x": float(line_words[0]["x0"]),
                    "y": float(line_words[0]["top"]),
                    "width": float(line_words[-1]["x1"]) - float(line_words[0]["x0"]),
                    "height": float(line_words[0]["bottom"]) - float(line_words[0]["top"]),
                    "bold": any(bold_flags),
                    "italic": any(italic_flags),
                    "fontname": line_words[0]["fontname"],
                    "page": page_number
                }

                parsed_blocks.append(block)

    return parsed_blocks

def load_all_blocks(data_dir="data/"):
    all_blocks = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                blocks = json.load(f)
                all_blocks.extend(blocks)

    # print(type(blocks[0]))  # should be <class 'dict'>
    return all_blocks

import json
import os
import difflib

def label_parsed_blocks(parsed_blocks, outline):
    labeled = []
    for block in parsed_blocks:
        block_text = block["text"].strip()
        block_page = block["page"]
        label = "None"

        # Try to match with outline items on same page
        for outline_item in outline:
            if outline_item["page"] != block_page:
                continue
            outline_text = outline_item["text"].strip()

            # Use fuzzy match — works better than strict equality
            if is_similar(block_text, outline_text):
                label = outline_item["level"]
                break

        block["label"] = label
        labeled.append(block)
    return labeled

def is_similar(text1, text2, threshold=0.45):
    ratio = difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    return ratio >= threshold

def label_dataset(parsed_dir, output_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(parsed_dir):
        if file.endswith(".json"):
            base = file.replace(".json", "")
            parsed_path = os.path.join(parsed_dir, file)
            output_path = os.path.join(output_dir, base + ".json")

            with open(parsed_path, "r") as f:
                parsed_blocks = json.load(f)

            with open(output_path, "r") as f:
                output = json.load(f)
                outline = output["outline"]

            labeled_blocks = label_parsed_blocks(parsed_blocks, outline)

            # Save labeled version
            with open(os.path.join(save_dir, base + "_labeled.json"), "w") as f:
                json.dump(labeled_blocks, f, indent=2)

            # Also write a new output JSON with the correct outline (from labeled_blocks)
            outline_new = [
                {"level": str(b["label"]), "text": str(b["text"]), "page": int(b["page"])}
                for b in labeled_blocks if b.get("label") and b["label"] != "None"
            ]
            output_new = {
                "title": output.get("title", base),
                "outline": outline_new
            }
            print(outline_new)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_new, f, indent=2, ensure_ascii=False)

from glob import glob
import json

# Load all labeled blocks
def load_labeled_blocks(dir_path):
    all_blocks = []
    for file in glob(f"{dir_path}/*.json"):
        with open(file) as f:
            blocks = json.load(f)
            all_blocks.extend(blocks)
    return all_blocks

if __name__ == "__main__":
    input_dir = "sample_dataset/pdfs"
    output_dir = "parsed_pdfs"
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".pdf"):
            input_pdf = os.path.join(input_dir, file)
            if os.path.exists(input_pdf):
                output_data = parse_pdf(input_pdf)
                output_json = os.path.splitext(file)[0] + ".json"
                output_path = os.path.join(output_dir, output_json)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"Parsed {len(output_data)} blocks from {input_pdf} -> {output_path}")
            else:
                print(f"[ERROR] File not found: {input_pdf}")

    label_dataset("parsed_pdfs", "/Users/sreenityathatikunta/Documents/Projects/adobe/output", "labeled_pdfs")
    blocks = load_labeled_blocks("labeled_pdfs/")  # This contains "label" field now
    X, raw_metadata = extract_features(blocks, page_height=800)
    y = [b["label"] for b in raw_metadata]

    # Train model
    train_model(X, y, "trained_model.pkl")