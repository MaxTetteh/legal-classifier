from flask import Flask, render_template, request
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import fitz  # PyMuPDF for PDF extraction
import requests
import os

#app = Flask(__name__)

# Hugging Face API settings
#HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
#HF_API_KEY = os.getenv("HF_API_KEY")  # set this in your PythonAnywhere env variables

#HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Candidate labels
CANDIDATE_LABELS = ["NDA", "SLA", "Employment", "Vendor", "Partnership"]

# Extractive summarizer
def summarize_text(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return " ".join([str(sentence) for sentence in summary])

# PDF extractor
def extract_text_from_pdf(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") or ""
    return text

# Hugging Face Zero-Shot Classification
def classify_text(text, candidate_labels):
    payload = {"sequence": text, "labels": candidate_labels}
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data["labels"][0]  # return top predicted class
    else:
        print("Error from HF API:", response.text)
        return "Unknown"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        input_text = ""

        # If PDF uploaded
        if "file" in request.files and request.files["file"].filename != "":
            pdf_file = request.files["file"]
            input_text = extract_text_from_pdf(pdf_file)

        # If text entered
        elif "text" in request.form:
            input_text = request.form["text"]

        if input_text.strip():
            # Summarize first
            summary = summarize_text(input_text)

            # Classify summary via Hugging Face API
            predicted_class = classify_text(summary, CANDIDATE_LABELS)

            result = {"predicted_class": predicted_class, "summary": summary}

    return render_template("index.html", result=result)

if __name__ == "__main__":
#    app.run(debug=True)
