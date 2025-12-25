import os
import json
import zipfile
import datetime
import calendar
import re
import random
import logging
import sys
import google.generativeai as genai
from collections import Counter, defaultdict
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tempfile
import shutil
import nltk
from nltk.corpus import stopwords
import emoji

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK Stopwords...")
    nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = os.urandom(24)

COLOR_MAP = {
    "DEFAULT": "#202124", "WHITE": "#202124", "RED": "#5C2B29", "ORANGE": "#614A19",
    "YELLOW": "#635D19", "GREEN": "#345920", "TEAL": "#16504B", "BLUE": "#2D555E",
    "DARK_BLUE": "#1E3A5F", "PURPLE": "#42275E", "PINK": "#5B2245", "BROWN": "#442F19",
    "GRAY": "#3C4043"
}

# --- PII & PRIVACY ---
EMAIL_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
PHONE_REGEX = r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
CREDIT_CARD_REGEX = r'\b(?:\d[ -]*?){13,16}\b'
SENSITIVE_KEYWORDS = ["password", "secret", "key", "ssn", "bank", "card", "pin", "medical", "doctor"]

def clean_pii(text):
    text = re.sub(EMAIL_REGEX, "[EMAIL REDACTED]", text)
    text = re.sub(PHONE_REGEX, "[PHONE REDACTED]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[CARD REDACTED]", text)
    return text

def is_content_safe(text):
    text_lower = text.lower()
    for kw in SENSITIVE_KEYWORDS:
        if kw in text_lower: return False
    return True

# --- ANALYSIS LOGIC ---

def analyze_year(extract_path, api_key=None, share_content=False):
    target_year = datetime.datetime.now().year
    # target_year = 2024 # Uncomment for testing
    
    logger.info(f"Starting analysis for year: {target_year}")
    
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(['http', 'https', 'www', 'com', 'google', 'null', 'true', 'false'])

    stats = {
        "year": target_year,
        
        # 1. Core Summary
        "total_created": 0,
        "total_edited": 0,
        "total_archived": 0,
        "total_trashed": 0,
        "first_note_date": None,
        "last_note_date": None,
        "oldest_active_note": None,
        
        # 2. Content
        "total_words": 0,
        "total_chars": 0,
        "word_buckets": {"Tiny (1-10)": 0, "Short (10-50)": 0, "Medium (50-200)": 0, "Long (200+)": 0},
        "emojis": Counter(),
        "common_words": Counter(),
        "starts_with": Counter(),
        "questions_count": 0,
        
        # 3. Labels
        "total_labels_used": 0,
        "notes_with_labels": 0,
        "notes_without_labels": 0,
        "label_counts": Counter(),
        "label_combinations": Counter(),
        
        # 4. Lists & Productivity
        "total_checklists": 0,
        "checklist_items_total": 0,
        "checklist_items_checked": 0,
        "checklist_verbs": Counter(),
        
        # 5. Media
        "total_images": 0,
        "total_audio": 0,
        "total_drawings": 0,
        "notes_with_media": 0,
        
        # 6. Time & Habits
        "heatmap": defaultdict(int),
        "days_of_week": Counter(),
        "hours_of_day": Counter(),
        "months": Counter(),
        "creation_dates": [],
        
        # 7. Lifecycle
        "revisited_notes": 0,
        "lifespans": [],
        
        # 8. AI/Fun/Dreams
        "longest_note": {"text": "", "len": 0, "date": ""},
        "dream_log": [], # Stores {type: 'Dream'|'Nightmare', text: '...', date: '...'}
        "content_samples": [],
        "aura": {"productive": 0, "creative": 0, "emotional": 0, "chaotic": 0},
        "ai_summary": {"archetype": "The Mystery", "description": "AI analysis skipped."}
    }

    files_processed = 0

    for root, dirs, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".json"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # Time Logic
                        c_ts = int(data.get("createdTimestampUsec", 0))
                        e_ts = int(data.get("userEditedTimestampUsec", 0))
                        
                        created_dt = datetime.datetime.fromtimestamp(c_ts / 1_000_000)
                        edited_dt = datetime.datetime.fromtimestamp(e_ts / 1_000_000)
                        
                        is_created_this_year = created_dt.year == target_year
                        is_edited_this_year = edited_dt.year == target_year
                        
                        if not is_edited_this_year and not is_created_this_year:
                            continue

                        # Core Counts
                        if is_created_this_year:
                            stats["total_created"] += 1
                            stats["heatmap"][created_dt.strftime("%Y-%m-%d")] += 1
                            stats["creation_dates"].append(created_dt.date())
                            
                            if stats["first_note_date"] is None or created_dt < stats["first_note_date"]:
                                stats["first_note_date"] = created_dt
                            if stats["last_note_date"] is None or created_dt > stats["last_note_date"]:
                                stats["last_note_date"] = created_dt
                            
                            stats["days_of_week"][calendar.day_name[created_dt.weekday()]] += 1
                            stats["hours_of_day"][created_dt.hour] += 1
                            stats["months"][created_dt.strftime("%B")] += 1

                        if is_edited_this_year:
                            stats["total_edited"] += 1
                            if not is_created_this_year:
                                stats["oldest_active_note"] = created_dt if stats["oldest_active_note"] is None or created_dt < stats["oldest_active_note"] else stats["oldest_active_note"]
                        
                        if data.get("isArchived"): stats["total_archived"] += 1
                        if data.get("isTrashed"): stats["total_trashed"] += 1

                        # Text Content
                        text_body = data.get("textContent", "")
                        list_items = data.get("listContent", [])
                        list_text = " ".join([item.get("text", "") for item in list_items])
                        full_text = (text_body + " " + list_text).strip()
                        
                        word_count = len(full_text.split())
                        stats["total_words"] += word_count
                        stats["total_chars"] += len(full_text)
                        
                        # Bucket Stats
                        if word_count <= 10: stats["word_buckets"]["Tiny (1-10)"] += 1
                        elif word_count <= 50: stats["word_buckets"]["Short (10-50)"] += 1
                        elif word_count <= 200: stats["word_buckets"]["Medium (50-200)"] += 1
                        else: stats["word_buckets"]["Long (200+)"] += 1
                        
                        # Emojis
                        for e in emoji.emoji_list(full_text):
                            stats["emojis"][e['emoji']] += 1
                            
                        # Questions
                        if "?" in full_text: stats["questions_count"] += 1

                        # Labels
                        labels = data.get("labels", [])
                        if labels:
                            stats["notes_with_labels"] += 1
                            stats["total_labels_used"] += len(labels)
                            names = sorted([l['name'] for l in labels])
                            for name in names: stats["label_counts"][name] += 1
                            if len(names) > 1: stats["label_combinations"][" + ".join(names[:2])] += 1
                        else:
                            stats["notes_without_labels"] += 1

                        # Checklists
                        if list_items:
                            stats["total_checklists"] += 1
                            stats["checklist_items_total"] += len(list_items)
                            checked = sum(1 for i in list_items if i.get("isChecked"))
                            stats["checklist_items_checked"] += checked
                            for item in list_items:
                                txt = item.get("text", "").strip()
                                if txt and txt.split()[0].isalpha():
                                    stats["checklist_verbs"][txt.split()[0].title()] += 1

                        # Media
                        has_media = False
                        if data.get("attachments"):
                            for att in data["attachments"]:
                                mime = att.get("mimetype", "")
                                if "image" in mime: stats["total_images"] += 1; has_media = True
                                elif "audio" in mime: stats["total_audio"] += 1; has_media = True
                                elif "drawing" in mime: stats["total_drawings"] += 1; has_media = True
                        if has_media: stats["notes_with_media"] += 1

                        # Lifecycle
                        delta = edited_dt - created_dt
                        if delta.days > 0: stats["lifespans"].append(delta.days)
                        if delta.days > 30: stats["revisited_notes"] += 1

                        # Longest Note
                        if len(full_text) > stats["longest_note"]["len"]:
                            stats["longest_note"] = {
                                "text": full_text[:400] + "..." if len(full_text) > 400 else full_text,
                                "len": len(full_text),
                                "date": created_dt.strftime("%B %d")
                            }

                        # --- DREAM & NIGHTMARE DETECTION ---
                        label_names = [l['name'].lower() for l in labels]
                        full_text_lower = full_text.lower()
                        
                        # Heuristics
                        is_nightmare = "nightmare" in label_names or "#nightmare" in full_text_lower
                        is_dream = "dream" in label_names or "#dream" in full_text_lower
                        
                        if len(full_text) > 20:
                            dream_type = None
                            if is_nightmare: dream_type = "Nightmare"
                            elif is_dream: dream_type = "Dream"
                            
                            if dream_type:
                                stats["dream_log"].append({
                                    "type": dream_type,
                                    "date": created_dt.strftime("%b %d"),
                                    "snippet": clean_pii(full_text[:200] + "...")
                                })

                        # Common Words
                        words = [w.lower() for w in full_text.split() if w.isalpha()]
                        clean_words = [w for w in words if w not in stop_words and len(w) > 3]
                        stats["common_words"].update(clean_words)

                        # Aura
                        if list_items: stats["aura"]["productive"] += 2
                        if has_media: stats["aura"]["creative"] += 3
                        if len(full_text) < 10: stats["aura"]["chaotic"] += 1
                        if full_text:
                            vs = analyzer.polarity_scores(full_text)
                            if vs['compound'] < -0.2: stats["aura"]["emotional"] += 2
                            
                            # Content Sampling (General)
                            if share_content and is_content_safe(full_text) and len(full_text) > 30 and not (is_dream or is_nightmare):
                                stats["content_samples"].append(clean_pii(full_text[:300]))

                        files_processed += 1

                except Exception as e:
                    logger.error(f"Error processing {file}", exc_info=True)

    # --- POST PROCESSING ---
    if stats["total_created"] == 0: return None

    # Streak
    unique_days = sorted(list(set(stats["creation_dates"])))
    streak = 0
    max_streak = 0
    for i in range(1, len(unique_days)):
        if (unique_days[i] - unique_days[i-1]).days == 1: streak += 1
        else: streak = 1
        max_streak = max(max_streak, streak)
    stats["streak"] = max_streak

    # Flatten
    stats["top_emojis"] = stats["emojis"].most_common(5)
    stats["top_words"] = stats["common_words"].most_common(15)
    stats["top_labels"] = stats["label_counts"].most_common(5)
    stats["top_verbs"] = stats["checklist_verbs"].most_common(5)
    stats["top_day"] = stats["days_of_week"].most_common(1)
    stats["top_hour"] = stats["hours_of_day"].most_common(1)
    
    # Ratios
    stats["label_rate"] = int((stats["notes_with_labels"] / stats["total_created"])*100) if stats["total_created"] else 0
    stats["completion_rate"] = int((stats["checklist_items_checked"] / stats["checklist_items_total"])*100) if stats["checklist_items_total"] else 0
    stats["avg_words"] = int(stats["total_words"] / stats["total_created"]) if stats["total_created"] else 0
    
    # Formats
    if stats["first_note_date"]: stats["first_note_date"] = stats["first_note_date"].strftime("%B %d")
    if stats["last_note_date"]: stats["last_note_date"] = stats["last_note_date"].strftime("%B %d")
    
    total_aura = sum(stats["aura"].values()) + 1
    stats["aura_norm"] = {k: int((v/total_aura)*100) for k,v in stats["aura"].items()}

    # --- GEMINI INTEGRATION ---
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt_context = ""
            
            # 1. Add General Note Samples
            if share_content and stats["content_samples"]:
                samples = "\n".join(random.sample(stats["content_samples"], min(10, len(stats["content_samples"]))))
                prompt_context += f"\n\nHere are random snippets from their general notes:\n{samples}"
            
            # 2. Add Dream/Nightmare Samples (Specific Feature)
            if share_content and stats["dream_log"]:
                # Grab a few dreams/nightmares
                dream_samples = [d['snippet'] for d in stats['dream_log'][:5]]
                prompt_context += f"\n\nHere are entries from their Dream/Nightmare Log (analyze their subconscious): \n" + "\n".join(dream_samples)

            prompt = f"""
            Analyze these Google Keep stats for a "Year in Review":
            - Total Notes: {stats['total_created']}
            - Checklist Completion: {stats['completion_rate']}%
            - Top Emojis: {stats['top_emojis']}
            - Top Labels: {stats['top_labels']}
            - Dreams Recorded: {len(stats['dream_log'])}
            - Vibe: Productive={stats['aura']['productive']}, Creative={stats['aura']['creative']}, Emotional={stats['aura']['emotional']}
            
            {prompt_context}
            
            1. Give them a creative "Archetype" name.
            2. Write a 2-sentence description of their brain/subconscious.
            
            Output format: ARCHETYPE: [Name] | DESCRIPTION: [Text]
            """
            
            resp = model.generate_content(prompt)
            txt = resp.text.strip()
            if "ARCHETYPE:" in txt:
                parts = txt.split('|')
                stats["ai_summary"] = {"archetype": parts[0].replace("ARCHETYPE:", "").strip(), "description": parts[1].replace("DESCRIPTION:", "").strip()}
            else:
                stats["ai_summary"]["description"] = txt

        except Exception as e:
            logger.error("Gemini API Error", exc_info=True)

    return stats

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        api_key = request.form.get('api_key', '').strip()
        share_content = request.form.get('share_content') == 'on'
        
        if file and file.filename.endswith('.zip'):
            temp_dir = tempfile.mkdtemp()
            try:
                zip_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                stats = analyze_year(temp_dir, api_key if api_key else None, share_content)
                if not stats: return render_template('index.html', error="No data found for this year.")
                
                return render_template('wrapped.html', stats=stats, color_map=COLOR_MAP)
            except Exception as e:
                logger.error("System Error", exc_info=True)
                return render_template('index.html', error="Error processing file.")
            finally:
                shutil.rmtree(temp_dir)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
