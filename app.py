import io
from flask import Flask, render_template, request
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import nltk
import fitz  # PyMuPDF
from PIL import Image

# Try to import pytesseract safely
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Ensure punkt tokenizer is available for Sumy
nltk.download("punkt", quiet=True)

app = Flask(__name__)

# Load Local Zero-Shot Model
MODEL_DIR = "./models/distilbart-mnli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Candidate labels
CANDIDATE_LABELS = [
    "Non-Disclosure Agreement",
    "Service Level Agreement",
    "Employment Agreement",
    "Partnership Agreement",
    "Vendor Agreement"
]

# Summarization Function
def extractive_summary(text, max_words=120):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary_sentences = summarizer(parser.document, sentences_count=10)

    summary_text = " ".join(str(sentence) for sentence in summary_sentences)
    words = summary_text.split()

    if len(words) > max_words:
        words = words[:max_words]

    return " ".join(words)

# PDF extractor (works with both text-based + scanned PDFs, no temp file writes)
def extract_text_from_pdf(file_storage):
    # Open PDF directly from memory buffer
    pdf_bytes = file_storage.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = ""
    for page_num in range(len(doc)):
        page = doc[page_num]

        # Try digital text extraction
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            # OCR fallback if available
            if OCR_AVAILABLE:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text += pytesseract.image_to_string(img)
            else:
                print(f"[Warning] No extractable text on page {page_num+1}, OCR not available.")
                text += " "  # keep page alignment

    doc.close()
    return text.strip()

# Routes
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
            # Summarize
            summary = extractive_summary(extracted_text)

            # Classify
            classification = classifier(summary, candidate_labels=CANDIDATE_LABELS)
            predicted_class = classification["labels"][0]

            result = {
                "predicted_class": predicted_class,
                "summary": summary
            }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
