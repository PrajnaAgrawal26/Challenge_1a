import os
import sys
import json
from pathlib import Path
from main import parse_pdf
from features import extract_features
from model import predict_headings

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")
MODEL_PATH = Path("trained_model.pkl")
PAGE_HEIGHT = 800  # You may want to estimate this per PDF

# Load model if exists
if not MODEL_PATH.exists():
    print("[ERROR] Model file not found. Please provide 'trained_model.pkl' in the build context.")
    sys.exit(1)

def extract_title(blocks):
    for block in blocks:
        if int(block.get("page", 0)) == 1 and block.get("text") and block["text"].strip():
            return block["text"].strip()
    return "Untitled"

def process_pdf(pdf_path):
    parsed_blocks = parse_pdf(str(pdf_path))
    X_new, raw_meta = extract_features(parsed_blocks, page_height=PAGE_HEIGHT)
    predicted_labels = predict_headings(str(MODEL_PATH), X_new)
    for i, label in enumerate(predicted_labels):
        raw_meta[i]["label"] = label
    title = extract_title(raw_meta)
    outline = [
        {"level": str(b["label"]), "text": str(b["text"]), "page": int(b["page"])}
        for b in raw_meta
    ]
    # print(f"Extracted title: {title}")
    # print(f"Extracted outline: {outline}")
    return {"title": title, "outline": outline}

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in /app/input.")
        return
    for pdf_file in pdf_files:
        result = process_pdf(pdf_file)
        output_file = OUTPUT_DIR / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Processed {pdf_file.name} -> {output_file.name}")

if __name__ == "__main__":
    main()
