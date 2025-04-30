from dataclasses import asdict
from flask import Flask, request, jsonify
from flask_cors import CORS
from app_types import GeneratedResponse
from dummies import get_sample_note
from generation_pipeline import run_generate
from keyword_extraction import KeywordExtractor
from summarizer import PegasusSummarizer 
import torch
torch.cuda.empty_cache()

app = Flask(__name__)


extractor = KeywordExtractor()
summarizer = PegasusSummarizer()

CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://1cz2hd3b-5173.asse.devtunnels.ms",
            "http://localhost:5173",
            "https://your-frontend.com"
        ]
    }
}, supports_credentials=True)

@app.route("/")
def health_check():
    return "healthy"

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        doc = data.get("document")
        if not doc:
            return jsonify({"error": "Missing 'document' field"}), 400

        response = run_generate(
            document=doc,
            extractor=extractor,
            summarizer=summarizer
        )
        return jsonify(response), 200

    except Exception as e:
        # Log the error if needed
        print(f"Error in /api/generate: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500