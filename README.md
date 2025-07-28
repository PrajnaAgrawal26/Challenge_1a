
# PDF Outline Extraction and Labeling

This project provides tools for parsing PDF files, extracting text blocks, labeling them according to a provided outline, and training a machine learning model to predict headings and structure in new PDFs. It is designed for document understanding and information extraction tasks, such as those found in document AI hackathons or research.

## Features
- **PDF Parsing:** Extracts text, font, position, and style metadata from PDF files using `pdfplumber`.
- **Block Labeling:** Matches parsed text blocks to a provided outline using fuzzy matching and assigns hierarchical labels.
- **Feature Extraction:** Converts labeled blocks into feature vectors for machine learning.
- **Model Training:** Trains a model to predict heading levels and structure from extracted features.
- **Batch Processing:** Supports processing and labeling of multiple PDFs in batch mode.

## Directory Structure
```
├── main.py                # Main script for parsing, labeling, and training
├── features.py            # Feature extraction logic
├── model.py               # Model training and prediction logic
├── process_pdfs.py        # (Optional) Additional PDF processing utilities
├── requirements.txt       # Python dependencies
├── sample_dataset/        # Example PDFs and outputs
│   ├── pdfs/              # Sample input PDFs
│   ├── outputs/           # Sample output JSONs
│   └── schema/            # Output schema definition
├── parsed_pdfs/           # Parsed blocks from PDFs (auto-generated)
├── labeled_pdfs/          # Labeled blocks with heading levels (auto-generated)
├── output/                # Output JSONs with outlines
├── trained_model.pkl      # Trained model file (auto-generated)
```

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Place your input PDFs in `sample_dataset/pdfs/`.
- Place the corresponding outline JSONs in `output/` (see sample format in `sample_dataset/outputs/`).

### 3. Run the Pipeline
```bash
python main.py
```
This will:
- Parse all PDFs in `sample_dataset/pdfs/` and save parsed blocks to `parsed_pdfs/`.
- Label the parsed blocks using outlines from `output/`, saving labeled blocks to `labeled_pdfs/`.
- Train a model using the labeled data and save it as `trained_model.pkl`.

### 4. Model Usage
You can use the trained model to predict headings/structure for new PDFs by importing the relevant functions from `model.py`.

## File Formats
- **Parsed Block JSON:** List of dicts with keys: `text`, `font_size`, `x`, `y`, `width`, `height`, `bold`, `italic`, `fontname`, `page`.
- **Labeled Block JSON:** Same as above, with an additional `label` field.
- **Outline JSON:**
  ```json
  {
    "title": "Document Title",
    "outline": [
      {"level": "1", "text": "Section Heading", "page": 1},
      ...
    ]
  }
  ```

## Requirements
- Python 3.7+
- pdfplumber
- scikit-learn
- numpy
- pandas
