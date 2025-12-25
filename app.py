import os
import json
import zipfile
import datetime
import calendar
import re
import random
import google.generativeai as genai
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile
import shutil
import nltk
from nltk.corpus import stopwords

# --- CONFIGURATION & SETUP ---

# Download NLTK data if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Colors for UI mapping
COLOR_MAP = {
    "DEFAULT": "#202124", "WHITE": "#202124", "RED": "#5C2B29", "ORANGE": "#614A19",
    "YELLOW": "#635D19", "GREEN": "#345920", "TEAL": "#16504B", "BLUE": "#2D555E",
    "DARK_BLUE": "#1E3A5F", "PURPLE": "#42275E", "PINK": "#5B2245", "BROWN": "#442F19",
    "GRAY": "#3C4043"
}

# --- PII & PRIVACY FILTERS ---

# Regex patterns for sensitive data
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_REGEX = r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
CREDIT_CARD_REGEX = r'\b(?:\d[ -]*?){13,16}\b'
# Keywords that flag a note as "unsafe" for AI sharing
SENSITIVE_KEYWORDS = [
    "password", "passwd", "secret", "key", "token", "ssn", "social security", 
    "bank", "account number", "routing", "credit card", "debit card", "pin", 
    "login", "medical", "prescription", "doctor", "diagnosis"
]

def clean_pii(text):
    """Redacts emails and phone numbers from text."""
    text = re.sub(EMAIL_REGEX, "[EMAIL REDACTED]", text)
    text = re.sub(PHONE_REGEX, "[PHONE REDACTED]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[CARD REDACTED]", text)
    return text

def is_content_safe(text):
    """Returns False if text contains sensitive keywords."""
    text_lower = text.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in text_lower:
            return False
    return True

# --- ANALYSIS LOGIC ---

def analyze_year(extract_path, api_key=None, share_content=False):
    target_year = datetime.datetime.now().year
    # target_year = 2024 # Uncomment for testing older zips
    
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['http', 'https', 'www', 'com', 'google', 'checked', 'false', 'true', 'null', 'item'])

    stats = {
        "year": target_year,
        "count": 0,
        "total_chars": 0,
        "book_equivalent": 0,
        "streak": 0,
        "dates_active": set(),
        "lists": 0,
        "images": 0,
        "voice": 0,
        "days": Counter(),
        "hours": Counter(),
        "colors": Counter(),
        "tags": Counter(),
        "topics": Counter(), 
        "common_words": Counter(),
        "sentiment_breakdown": {"pos": 0, "neg": 0, "neu": 0},
        "longest_note": {"text": "", "len": 0, "date": ""},
        "dreams": [],
        "ai_summary": {"archetype": "The Mystery", "description": "AI analysis was skipped or failed."},
        "aura": {"productive": 0, "creative": 0, "emotional": 0, "chaotic": 0},
        "content_samples": [] # Temporarily store safe notes for AI
    }

    notes_processed = 0

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Timestamp handling
                        ts_usec = int(data.get("userEditedTimestampUsec", data.get("createdTimestampUsec", 0)))
                        dt = datetime.datetime.fromtimestamp(ts_usec / 1_000_000)
                        
                        if dt.year != target_year or data.get("isTrashed"): continue

                        # Basic Metrics
                        stats["count"] += 1
                        stats["dates_active"].add(dt.date())
                        stats["days"][calendar.day_name[dt.weekday()]] += 1
                        stats["hours"][dt.hour] += 1
                        stats["colors"][data.get("color", "DEFAULT")] += 1

                        # Text Construction
                        text_content = data.get("textContent", "")
                        list_content = ""
                        is_list = False
                        
                        if data.get("listContent"):
                            is_list = True
                            stats["lists"] += 1
                            for item in data["listContent"]:
                                list_content += " " + item.get("text", "")
                        
                        full_text = (text_content + " " + list_content).strip()
                        char_len = len(full_text)
                        stats["total_chars"] += char_len

                        # Attachments
                        if data.get("attachments"):
                            for att in data["attachments"]:
                                if "image" in att.get("mimetype", ""): stats["images"] += 1
                                if "audio" in att.get("mimetype", ""): stats["voice"] += 1

                        # Longest Note
                        if char_len > stats["longest_note"]["len"]:
                            stats["longest_note"] = {
                                "text": full_text[:400] + "..." if char_len > 400 else full_text,
                                "len": char_len,
                                "date": dt.strftime("%B %d")
                            }

                        # Dream Detection
                        labels = [l['name'].lower() for l in data.get("labels", [])]
                        is_dream = False
                        if "dream" in labels or "nightmare" in labels or "#dream" in full_text.lower():
                            is_dream = True
                        
                        if is_dream and len(full_text) > 20:
                            stats["dreams"].append({
                                "date": dt.strftime("%b %d"),
                                "snippet": full_text[:150] + "..."
                            })

                        # Tags & Topics
                        for l in data.get("labels", []):
                            name = l['name']
                            stats["tags"][name] += 1
                            # Heuristics
                            nl = name.lower()
                            if nl in ['gym', 'workout', 'health', 'food', 'diet']: stats["topics"]['Health & Body'] += 1
                            elif nl in ['code', 'dev', 'work', 'job', 'meeting', 'to-do']: stats["topics"]['The Grind'] += 1
                            elif nl in ['movie', 'book', 'game', 'music', 'watch']: stats["topics"]['Media'] += 1
                            elif nl in ['idea', 'art', 'write', 'poetry']: stats["topics"]['Creativity'] += 1
                            elif nl in ['finance', 'money', 'budget', 'bill']: stats["topics"]['Finance'] += 1

                        # NLP Analysis
                        if full_text:
                            # Sentiment
                            vs = analyzer.polarity_scores(full_text)
                            compound = vs['compound']
                            if compound >= 0.05: stats["sentiment_breakdown"]["pos"] += 1
                            elif compound <= -0.05: stats["sentiment_breakdown"]["neg"] += 1
                            else: stats["sentiment_breakdown"]["neu"] += 1

                            # Word Frequency
                            words = [w.lower() for w in full_text.split() if w.isalpha()]
                            clean_words = [w for w in words if w not in stop_words and len(w) > 3]
                            stats["common_words"].update(clean_words)

                            # Aura
                            if is_list: stats["aura"]["productive"] += 2
                            elif compound < -0.3: stats["aura"]["emotional"] += 3
                            elif char_len > 500: stats["aura"]["creative"] += 4
                            elif char_len < 10: stats["aura"]["chaotic"] += 1

                            # Sample Collection for AI (if safe)
                            if share_content and is_content_safe(full_text) and len(full_text) > 20:
                                stats["content_samples"].append(clean_pii(full_text[:300])) # Limit per note length

                except Exception as e:
                    print(f"Skipped file: {e}")

    if stats["count"] == 0: return None

    # Streak Calculation
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

    stats["book_equivalent"] = round(stats["total_chars"] / 30000, 1)
    stats["top_words"] = stats["common_words"].most_common(25)
    stats["top_tags"] = stats["tags"].most_common(8)
    stats["top_topics"] = stats["topics"].most_common(3)
    stats["top_days"] = stats["days"].most_common(1)
    
    total_aura = sum(stats["aura"].values()) + 1
    stats["aura_norm"] = {k: int((v/total_aura)*100) for k,v in stats["aura"].items()}

    # --- GEMINI INTEGRATION ---
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = ""
            
            # Scenario A: Deep Analysis (Content Shared)
            if share_content and stats["content_samples"]:
                # Limit total payload to avoid token limits (approx 15 notes)
                sample_notes = random.sample(stats["content_samples"], min(15, len(stats["content_samples"])))
                notes_text = "\n---\n".join(sample_notes)
                
                prompt = f"""
                Analyze these random snippets from a user's Google Keep notes:
                "{notes_text}"
                
                And these stats:
                - Total Notes: {stats['count']}
                - Top Tags: {', '.join([t[0] for t in stats['top_tags']])}
                - Most active time: {stats['top_days'][0][0]}s
                
                Based on this, create a "Spotify Wrapped" style personality profile.
                1. Assign a creative "Archetype Name" (e.g., The Midnight Poet, The Chaos Coordinator).
                2. Write a witty, slightly roast-y description of their year (max 3 sentences).
                
                Output ONLY in this format:
                ARCHETYPE: [Name] | DESCRIPTION: [Text]
                """
            
            # Scenario B: Shallow Analysis (Stats Only)
            else:
                prompt = f"""
                Act as a Gen-Z data analyst. Based on these note-taking stats, assign a user archetype.
                - Total Notes: {stats['count']}
                - Vibe: {'Productive' if stats['aura']['productive'] > stats['aura']['emotional'] else 'Emotional'}
                - Top Words: {', '.join([w[0] for w in stats['top_words'][:5]])}
                - Top Tags: {', '.join([t[0] for t in stats['top_tags']])}
                - Streak: {stats['streak']} days
                
                1. Assign a creative "Archetype Name".
                2. Write a witty description (max 3 sentences).
                
                Output ONLY in this format:
                ARCHETYPE: [Name] | DESCRIPTION: [Text]
                """
            
            response = model.generate_content(prompt)
            text_resp = response.text.strip()
            
            if "ARCHETYPE:" in text_resp:
                parts = text_resp.split('|')
                stats["ai_summary"] = {
                    "archetype": parts[0].replace("ARCHETYPE:", "").strip(),
                    "description": parts[1].replace("DESCRIPTION:", "").strip() if len(parts) > 1 else ""
                }
            else:
                stats["ai_summary"]["description"] = text_resp

        except Exception as e:
            print(f"Gemini Error: {e}")
            stats["ai_summary"]["description"] = "AI was too stunned to speak (Error or Quota exceeded)."

    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        api_key = request.form.get('api_key', '').strip()
        share_content = request.form.get('share_content') == 'on' # Checkbox check
        
        if file and file.filename.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                stats = analyze_year(temp_dir, api_key if api_key else None, share_content)
                
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
