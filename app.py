import os
import json
import zipfile
import datetime
import calendar
import google.generativeai as genai
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile
import shutil
import nltk
from nltk.corpus import stopwords

# NLTK Setup (Run once on import)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

COLOR_MAP = {
    "DEFAULT": "#202124", "WHITE": "#202124", "RED": "#5C2B29", "ORANGE": "#614A19",
    "YELLOW": "#635D19", "GREEN": "#345920", "TEAL": "#16504B", "BLUE": "#2D555E",
    "DARK_BLUE": "#1E3A5F", "PURPLE": "#42275E", "PINK": "#5B2245", "BROWN": "#442F19",
    "GRAY": "#3C4043"
}

def analyze_year(extract_path, api_key=None):
    target_year = datetime.datetime.now().year
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    # Add common note noise to stopwords
    stop_words.update(['http', 'https', 'www', 'com', 'google', 'checked', 'false', 'true'])

    stats = {
        "year": target_year,
        "count": 0,
        "total_chars": 0,
        "book_equivalent": 0, # Assuming 50k chars is a novella
        "streak": 0,
        "dates_active": set(),
        "lists": 0,
        "images": 0,
        "voice": 0,
        "pinned": 0,
        "days": Counter(),
        "hours": Counter(),
        "colors": Counter(),
        "tags": Counter(),
        "topics": Counter(), # Heuristic topics
        "common_words": Counter(),
        "sentiment_breakdown": {"pos": 0, "neg": 0, "neu": 0},
        "longest_note": {"text": "", "len": 0, "date": ""},
        "dreams": [], # Store dream snippets
        "ai_summary": "AI generation skipped (No API Key provided).",
        "aura": {"productive": 0, "creative": 0, "emotional": 0, "chaotic": 0}
    }

    notes_processed = 0

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Filter Year
                        ts_usec = int(data.get("userEditedTimestampUsec", data.get("createdTimestampUsec", 0)))
                        dt = datetime.datetime.fromtimestamp(ts_usec / 1_000_000)
                        
                        if dt.year != target_year or data.get("isTrashed"): continue

                        # Metadata
                        stats["count"] += 1
                        stats["dates_active"].add(dt.date())
                        stats["days"][calendar.day_name[dt.weekday()]] += 1
                        stats["hours"][dt.hour] += 1
                        stats["colors"][data.get("color", "DEFAULT")] += 1
                        if data.get("isPinned"): stats["pinned"] += 1

                        # Content Construction
                        text = data.get("textContent", "")
                        list_text = ""
                        is_list = False
                        if data.get("listContent"):
                            is_list = True
                            stats["lists"] += 1
                            for item in data["listContent"]:
                                list_text += " " + item.get("text", "")
                        
                        full_text = (text + " " + list_text).strip()
                        char_len = len(full_text)
                        stats["total_chars"] += char_len

                        # Images/Audio
                        if data.get("attachments"):
                            for att in data["attachments"]:
                                if "image" in att.get("mimetype", ""): stats["images"] += 1
                                if "audio" in att.get("mimetype", ""): stats["voice"] += 1

                        # Longest Note Tracker
                        if char_len > stats["longest_note"]["len"]:
                            stats["longest_note"] = {
                                "text": full_text[:400] + "..." if char_len > 400 else full_text,
                                "len": char_len,
                                "date": dt.strftime("%B %d")
                            }

                        # Dream Detection (Tag or Keyword)
                        is_dream = False
                        labels = [l['name'].lower() for l in data.get("labels", [])]
                        if "dream" in labels or "nightmare" in labels or "#dream" in full_text.lower():
                            is_dream = True
                        
                        # Only add to dream log if it's substantial
                        if is_dream and len(full_text) > 20:
                            stats["dreams"].append({
                                "date": dt.strftime("%b %d"),
                                "snippet": full_text[:150] + "..."
                            })

                        # Tags
                        for l in data.get("labels", []):
                            stats["tags"][l['name']] += 1
                            # Heuristic Topic Mapping
                            name = l['name'].lower()
                            if name in ['gym', 'workout', 'health', 'food']: stats["topics"]['Health & Body'] += 1
                            elif name in ['code', 'python', 'dev', 'work', 'job']: stats["topics"]['Grind & Tech'] += 1
                            elif name in ['movie', 'book', 'game', 'music']: stats["topics"]['Media Consumption'] += 1
                            elif name in ['idea', 'inspiration', 'art']: stats["topics"]['Creative Sparks'] += 1

                        # NLP & VADER
                        if full_text:
                            # Sentiment
                            vs = analyzer.polarity_scores(full_text)
                            compound = vs['compound']
                            if compound >= 0.05: stats["sentiment_breakdown"]["pos"] += 1
                            elif compound <= -0.05: stats["sentiment_breakdown"]["neg"] += 1
                            else: stats["sentiment_breakdown"]["neu"] += 1

                            # Word Freq
                            words = [w.lower() for w in full_text.split() if w.isalpha()]
                            clean_words = [w for w in words if w not in stop_words and len(w) > 3]
                            stats["common_words"].update(clean_words)

                            # Aura Scoring
                            if is_list: stats["aura"]["productive"] += 2
                            elif compound < -0.3: stats["aura"]["emotional"] += 3
                            elif char_len > 500: stats["aura"]["creative"] += 4
                            elif char_len < 10: stats["aura"]["chaotic"] += 1

                except Exception as e:
                    print(f"Skipped: {e}")

    # Post Processing
    if stats["count"] == 0: return None

    # Streak Calc
    dates = sorted(list(stats["dates_active"]))
    longest_streak = 0
    current_streak = 0
    for i in range(len(dates)):
        if i == 0: 
            current_streak = 1
        else:
            delta = dates[i] - dates[i-1]
            if delta.days == 1:
                current_streak += 1
            elif delta.days > 1:
                longest_streak = max(longest_streak, current_streak)
                current_streak = 1
    stats["streak"] = max(longest_streak, current_streak)

    stats["book_equivalent"] = round(stats["total_chars"] / 30000, 1) # ~30k chars is a minimal novella
    stats["top_words"] = stats["common_words"].most_common(25)
    stats["top_tags"] = stats["tags"].most_common(8)
    stats["top_topics"] = stats["topics"].most_common(3)
    stats["top_days"] = stats["days"].most_common(1)
    
    # Normalize Aura
    total_aura = sum(stats["aura"].values()) + 1
    stats["aura_norm"] = {k: int((v/total_aura)*100) for k,v in stats["aura"].items()}

    # --- GEMINI INTEGRATION ---
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Privacy: We only send stats, not actual note content (except standard tags)
            prompt = f"""
            Act as a witty, Gen-Z data analyst writing a 'Spotify Wrapped' summary for a user's notes.
            Here are their stats for the year:
            - Total Notes: {stats['count']}
            - Main Vibe: {'Productive' if stats['aura']['productive'] > stats['aura']['emotional'] else 'Emotional'}
            - Top Words: {', '.join([w[0] for w in stats['top_words'][:5]])}
            - Top Tags: {', '.join([t[0] for t in stats['top_tags']])}
            - Longest Streak: {stats['streak']} days
            - Total Characters: {stats['total_chars']}
            
            Write a short paragraph (max 3 sentences) diagnosing their personality based on this. Be funny but insightful.
            Then, give them a specific "User Archetype" name (e.g., "The Midnight Poet").
            Format output as: ARCHETYPE: [Name] | DESCRIPTION: [Text]
            """
            
            response = model.generate_content(prompt)
            stats["ai_summary"] = response.text
        except Exception as e:
            stats["ai_summary"] = "AI unavailable (API Key Error or Quota Exceeded)."

    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        api_key = request.form.get('api_key', '').strip()
        
        if file and file.filename.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                stats = analyze_year(temp_dir, api_key if api_key else None)
                
                if not stats:
                    return render_template('index.html', error="No notes found for this year.")
                
                return render_template('wrapped.html', stats=stats, color_map=COLOR_MAP)
            except Exception as e:
                print(e)
                return render_template('index.html', error="Error processing file.")
            finally:
                shutil.rmtree(temp_dir)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
