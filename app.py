import os
import json
import zipfile
import datetime
import calendar
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for
from textblob import TextBlob
from werkzeug.utils import secure_filename
import tempfile
import shutil

app = Flask(__name__)
app.secret_key = 'change_this_secret_key'

# Google Keep Color Hex Codes
COLOR_MAP = {
    "DEFAULT": "#202124", "WHITE": "#202124", "RED": "#5C2B29", "ORANGE": "#614A19",
    "YELLOW": "#635D19", "GREEN": "#345920", "TEAL": "#16504B", "BLUE": "#2D555E",
    "DARK_BLUE": "#1E3A5F", "PURPLE": "#42275E", "PINK": "#5B2245", "BROWN": "#442F19",
    "GRAY": "#3C4043"
}

def get_time_of_day(hour):
    if 5 <= hour < 12: return "Morning"
    elif 12 <= hour < 17: return "Afternoon"
    elif 17 <= hour < 22: return "Evening"
    else: return "Night Owl"

def analyze_year(extract_path):
    target_year = datetime.datetime.now().year
    # target_year = 2024 # Uncomment to force a specific year for testing

    stats = {
        "year": target_year,
        "count": 0,
        "words": 0,
        "lists": 0,
        "images": 0,
        "pinned": 0,
        "archived": 0,
        "days": Counter(),       # Mon, Tue...
        "hours": Counter(),      # 0-23
        "time_of_day": Counter(), # Morning/Night
        "months": Counter(),
        "colors": Counter(),
        "tags": Counter(),
        "interests": Counter(),  # Noun phrases
        "sentiment": {"pos": 0, "neg": 0, "neu": 0, "total_score": 0},
        "aura": {"productive": 0, "creative": 0, "emotional": 0, "clutter": 0},
        "top_positive_note": {"text": "", "score": -1},
        "top_negative_note": {"text": "", "score": 1}
    }

    notes_processed = 0

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 1. Filter by Year (User Edited Timestamp)
                        # Keep uses microseconds
                        ts_usec = int(data.get("userEditedTimestampUsec", data.get("createdTimestampUsec", 0)))
                        ts = ts_usec / 1_000_000
                        dt = datetime.datetime.fromtimestamp(ts)

                        if dt.year != target_year:
                            continue
                        
                        if data.get("isTrashed"): continue

                        # --- Basic Counting ---
                        stats["count"] += 1
                        notes_processed += 1
                        stats["pinned"] += 1 if data.get("isPinned") else 0
                        stats["archived"] += 1 if data.get("isArchived") else 0
                        stats["colors"][data.get("color", "DEFAULT")] += 1
                        
                        # --- Temporal Analysis ---
                        stats["days"][calendar.day_name[dt.weekday()]] += 1
                        stats["hours"][dt.hour] += 1
                        stats["time_of_day"][get_time_of_day(dt.hour)] += 1
                        stats["months"][dt.strftime("%B")] += 1

                        # --- Content Extraction ---
                        text_content = data.get("textContent", "")
                        list_content = ""
                        
                        is_list = False
                        if data.get("listContent"):
                            is_list = True
                            stats["lists"] += 1
                            for item in data["listContent"]:
                                list_content += " " + item.get("text", "")
                                # Check completed items for productivity score
                                if item.get("isChecked"): stats["aura"]["productive"] += 1

                        full_text = (text_content + list_content).strip()
                        stats["words"] += len(full_text.split())

                        # Images/Audio
                        has_image = False
                        if data.get("attachments"):
                            for att in data["attachments"]:
                                if "image" in att.get("mimetype", ""):
                                    stats["images"] += 1
                                    has_image = True

                        # Labels/Tags
                        if data.get("labels"):
                            for label in data["labels"]:
                                stats["tags"][label["name"]] += 1

                        # --- NLP Analysis (TextBlob) ---
                        if full_text:
                            blob = TextBlob(full_text)
                            sentiment = blob.sentiment.polarity
                            subjectivity = blob.sentiment.subjectivity
                            
                            stats["sentiment"]["total_score"] += sentiment
                            if sentiment > 0.1: stats["sentiment"]["pos"] += 1
                            elif sentiment < -0.1: stats["sentiment"]["neg"] += 1
                            else: stats["sentiment"]["neu"] += 1

                            # Track best/worst notes (store snippet)
                            if sentiment > stats["top_positive_note"]["score"] and len(full_text) > 20:
                                stats["top_positive_note"] = {"text": full_text[:150] + "...", "score": sentiment}
                            if sentiment < stats["top_negative_note"]["score"] and len(full_text) > 20:
                                stats["top_negative_note"] = {"text": full_text[:150] + "...", "score": sentiment}

                            # Interests (Noun Phrases) - Filter garbage
                            for phrase in blob.noun_phrases:
                                if len(phrase) > 4 and "http" not in phrase:
                                    stats["interests"][phrase.lower()] += 1

                            # --- Aura/Vibe Scoring ---
                            # Productive: Lists, short text, numbers
                            if is_list: stats["aura"]["productive"] += 2
                            
                            # Creative: Images, colors other than default/white, high word count
                            if has_image: stats["aura"]["creative"] += 5
                            if data.get("color") not in ["DEFAULT", "WHITE"]: stats["aura"]["creative"] += 1
                            
                            # Emotional: High subjectivity, long paragraphs
                            if subjectivity > 0.5 and not is_list: stats["aura"]["emotional"] += 3
                            
                            # Clutter: Unchecked items, very short notes
                            if len(full_text) < 5: stats["aura"]["clutter"] += 1

                except Exception as e:
                    print(f"Skipped file: {e}")

    # --- Post Processing ---
    if stats["count"] == 0:
        return None # Handle empty year

    stats["avg_sentiment"] = stats["sentiment"]["total_score"] / stats["count"]
    
    # Get top items
    stats["top_days"] = stats["days"].most_common(1)
    stats["top_time"] = stats["time_of_day"].most_common(1)
    stats["top_interests"] = stats["interests"].most_common(15)
    stats["top_tags"] = stats["tags"].most_common(10)
    
    # Normalize Aura for Chart (Scale 0-100 approx)
    aura_total = sum(stats["aura"].values()) + 1
    stats["aura_normalized"] = {k: int((v/aura_total)*100) for k,v in stats["aura"].items()}

    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        
        if file and file.filename.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                stats = analyze_year(temp_dir)
                
                if not stats:
                    return render_template('index.html', error="No notes found for the current year!")

                return render_template('wrapped.html', stats=stats, color_map=COLOR_MAP)
            except Exception as e:
                print(e)
                return render_template('index.html', error="Error processing file.")
            finally:
                shutil.rmtree(temp_dir)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
