import os
import json
import zipfile
import datetime
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash
from textblob import TextBlob
from werkzeug.utils import secure_filename
import tempfile
import shutil

app = Flask(__name__)

# Google Keep Color Mapping (Approximate Hex)
COLOR_MAP = {
    "DEFAULT": "#FFFFFF", "WHITE": "#FFFFFF", "RED": "#F28B82", "ORANGE": "#FBBC04",
    "YELLOW": "#FFF475", "GREEN": "#CCFF90", "TEAL": "#A7FFEB", "BLUE": "#CBF0F8",
    "DARK_BLUE": "#AECBFA", "PURPLE": "#D7AEFB", "PINK": "#FDCFE8", "BROWN": "#E6C9A8",
    "GRAY": "#E8EAED"
}

def analyze_notes(extract_path):
    notes = []
    stats = {
        "total_notes": 0,
        "total_lists": 0,
        "total_images": 0,
        "total_voice": 0,
        "oldest_date": None,
        "newest_date": None,
        "char_count": 0,
        "word_cloud": Counter(),
        "colors": Counter(),
        "timeline": Counter(),  # Notes per month
        "sentiment_score": 0,   # Aggregate sentiment
        "labels": Counter()
    }

    # Iterate through extracted files
    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Filter out non-note JSONs (metadata)
                        if "isTrashed" in data and data["isTrashed"]:
                            continue
                        
                        stats["total_notes"] += 1
                        stats["colors"][data.get("color", "DEFAULT")] += 1
                        
                        # Timestamps (Google uses microseconds)
                        ts = int(data.get("createdTimestampUsec", 0)) / 1000000
                        date_obj = datetime.datetime.fromtimestamp(ts)
                        month_key = date_obj.strftime("%Y-%m")
                        stats["timeline"][month_key] += 1
                        
                        if stats["oldest_date"] is None or date_obj < stats["oldest_date"]:
                            stats["oldest_date"] = date_obj
                        if stats["newest_date"] is None or date_obj > stats["newest_date"]:
                            stats["newest_date"] = date_obj

                        # Content Analysis
                        content = data.get("textContent", "")
                        
                        # Handle Checklists
                        if data.get("listContent"):
                            stats["total_lists"] += 1
                            for item in data["listContent"]:
                                content += " " + item.get("text", "")
                        
                        # Attachments
                        if data.get("attachments"):
                            for att in data["attachments"]:
                                if "image" in att.get("mimetype", ""):
                                    stats["total_images"] += 1
                                if "audio" in att.get("mimetype", ""):
                                    stats["total_voice"] += 1
                        
                        # Labels
                        if data.get("labels"):
                            for label in data["labels"]:
                                stats["labels"][label["name"]] += 1

                        # Text Stats
                        stats["char_count"] += len(content)
                        blob = TextBlob(content)
                        stats["sentiment_score"] += blob.sentiment.polarity
                        
                        # Basic Word Frequency (filtering short words)
                        words = [w.lower() for w in blob.words if len(w) > 3 and w.isalpha()]
                        stats["word_cloud"].update(words)

                except Exception as e:
                    print(f"Error parsing {file}: {e}")

    # Final Polish
    if stats["total_notes"] > 0:
        stats["sentiment_score"] = stats["sentiment_score"] / stats["total_notes"]
    
    # Sort Timeline properly
    stats["timeline"] = dict(sorted(stats["timeline"].items()))
    
    # Format dates
    if stats["oldest_date"]:
        stats["oldest_date"] = stats["oldest_date"].strftime("%B %d, %Y")
    
    # Get top 50 words
    stats["word_cloud"] = stats["word_cloud"].most_common(50)
    stats["colors"] = dict(stats["colors"])
    stats["labels"] = dict(stats["labels"].most_common(10))

    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.zip'):
            # Create a temp directory
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Analyze
                stats = analyze_notes(temp_dir)
                
                return render_template('wrapped.html', stats=stats, color_map=COLOR_MAP)
            finally:
                # Cleanup
                shutil.rmtree(temp_dir)
                
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
