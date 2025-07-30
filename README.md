# 🧠 Hybrid ATS Resume Analyzer

An advanced resume and job description analyzer that uses a hybrid scoring model (TF-IDF, Semantic & Keyword) to provide a comprehensive match score and actionable feedback. Built with PyTorch and Streamlit.

---

## 📂 Project Structure

```
.
├── Advanced.py                    # Main Streamlit application
├── domain_keywords.py             # Domain-specific keywords and categories
├── models/                        # Directory for downloaded AI models
├── datasets/                      # Directory for training datasets
├── resume_versions/               # Directory for saved resume versions
├── requirements.txt               # List of all Python dependencies
└── README.md                      # Project documentation
```

---

## ✨ Key Analysis Features

- **Hybrid Match Scoring**: Calculates a weighted score from TF-IDF, Semantic Similarity, and Keyword Coverage.
- **Keyword Analysis**: Identifies matched and missing keywords between the resume and job description.
- **Bullet Point Quality**: Scores experience points on the use of strong action verbs and quantification.
- **Readability & Sanity Checks**: Assesses resume readability and checks for contact information.
- **AI Content Generation**: Creates tailored cover letter drafts and interview questions using a local generative model.
- **Live Resume Editing**: Provides an interactive editor for making real-time improvements.
- **ATS Formatting Check**: Scans `.docx` and `.pdf` files for formatting that can confuse older ATS systems.

---

## ⚙️ Model Details

This project leverages multiple locally-run models, downloaded automatically on first setup.

### Semantic Analysis Models

- `TechWolf/JobBERT-v3`
- `sentence-transformers/all-MiniLM-L6-v2`
- `nhanv/cv_parser`

### Generative Model

- `google/flan-t5-base` (for interview questions, cover letters, etc.)

### Custom Training Base

- `distilroberta-base`

### Framework

- PyTorch

---

## 🖼️ Web App (Streamlit)

- Upload a resume and job description from a file or paste text directly.
- Receive a detailed match score and a breakdown of how it was calculated.
- Get real-time feedback and AI-powered suggestions for improvement.
- Use the "Batch Compare" tab to analyze multiple documents at once.

### ▶️ Run the App

```bash
streamlit run Advanced.py
```

### 📈 Sample Output

```
✅ Final Match Score: 88.45%
💡 Score Insight: "High semantic match in 'Experience' section contributed +35 points."
❌ Actionable Feedback: "Suggests rewriting 2 weak bullet points to include more quantification."
```

---

## 🔬 Model Fine-Tuning (Optional)

Use the "Train Model" tab to fine-tune a `distilroberta-base` model on your own dataset.

1. Place your `.csv` or `.jsonl` files in the `./datasets` directory.
2. Configure data loading options in the UI.
3. The newly trained model is saved to `./fine_tuned_hybrid_model` and becomes available for selection in the sidebar.

---

## 🔧 Requirements

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

A comprehensive `requirements.txt` would look like this:

```
streamlit
torch
pandas
scikit-learn
sentence-transformers
nltk
yake
spacy
python-docx
pypdf
streamlit-ace
huggingface-hub
pyresparser
textract
altair
textstat
```

> Note: After installation, you must also download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## 📸 Dataset

The app can automatically download a sample dataset for fine-tuning from [`jacob-hugging-face/job-descriptions`](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions) on Hugging Face. You can also add your own data to the `./datasets` folder.

---

## 👤 Author

* **C B Vinay**
* GitHub: [cbvinay-7](https://github.com/cbvinay-7)

---

## 🛡️ License

MIT License

---

## 📬 Contact

For suggestions or collaboration, feel free to reach out.
