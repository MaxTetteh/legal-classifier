import os
from flask import Flask, render_template, request
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from huggingface_hub import InferenceClient

# Ensure nltk punkt is available
nltk.download("punkt", quiet=True)

app = Flask(__name__)


# Zero-Shot Model (Hugging Face Hub, not local)
classifier = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-1"
)

# Candidate labels for classification
CANDIDATE_LABELS = [
    "Non-Disclosure Agreement",
    "Service Level Agreement",
    "Employment Agreement",
    "Vendor Agreement",
    "Partnership Agreement"
]


# Summarization function: with sumy lexrank
def extractive_summary(text, max_words=120):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=10)

    summary_text = " ".join(str(sentence) for sentence in summary_sentences)
    words = summary_text.split()

    if len(words) > max_words:
        words = words[:max_words]

    return " ".join(words)


# PDF Text Extractor (OCR enabled functionality for scanned PDFs)
def extract_text_from_pdf(file_storage):
    doc = fitz.open(stream=file_storage.read(), filetype="pdf")
    text = ""

    for page_num in range(len(doc)):
        page = doc[page_num]

         
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            # OCR fallback (for scanned PDFs)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img)

    return text


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        text_input = request.form.get("text", "").strip()
        file = request.files.get("file")

        extracted_text = ""

        if file and file.filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(file)
        elif text_input:
            extracted_text = text_input

        if extracted_text:
            summary = extractive_summary(extracted_text)

            classification = classifier(
                summary,
                candidate_labels=CANDIDATE_LABELS
            )

            predicted_class = classification["labels"][0]

            result = {
                "predicted_class": predicted_class,
                "summary": summary
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
