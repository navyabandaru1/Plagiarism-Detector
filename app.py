from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from bs4 import BeautifulSoup
from googlesearch import search
import requests

app = Flask(__name__)

def scrape_top_article(query):
    try:
        # Search Google and take the top result
        urls = list(search(query, num_results=1))
        if not urls:
            return ""

        url = urls[0]
        print("🔗 Scraping:", url)

        # Get article content
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract all <p> tags (paragraphs)
        paragraphs = soup.find_all('p')
        text = " ".join(p.get_text() for p in paragraphs)

        return text
    except Exception as e:
        print("❌ Error scraping:", e)
        return ""

# Clean the input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Get cosine similarity score using TF-IDF
def get_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(similarity * 100, 2)

# Plagiarism checker logic (fixed - no recursion)
def check_plagiarism(user_text):
    # Use the first 100 characters as the search query
    query = user_text[:100]

    # Scrape web content
    scraped_text = scrape_top_article(query)

    if not scraped_text.strip():
        return 0.0  # fallback if scraping fails

    user_text_clean = clean_text(user_text)
    scraped_text_clean = clean_text(scraped_text)

    score = get_similarity(user_text_clean, scraped_text_clean)
    print("Plagiarism Score:", score)
    return score

# Verdict generator based on score
def get_verdict(score):
    if score >= 75:
        return "❌ High plagiarism detected. Please revise your content."
    elif score >= 40:
        return "⚠️ Moderate plagiarism. Consider rephrasing some parts."
    elif score >= 20:
        return "🟡 Low plagiarism. Looks mostly original with minor matches."
    else:
        return "✅ Very low or no plagiarism detected. Good to go!"

@app.route('/')
def home():
    return render_template('index.html')  # Your HTML form page

@app.route('/check', methods=['POST'])
def check():
    user_text = ""
    if 'file' in request.files:
        uploaded_file = request.files['file']
        if uploaded_file and uploaded_file.filename.endswith('.txt'):
            user_text = uploaded_file.read().decode('utf-8')

    if not user_text:
        user_text = request.form.get('text', '')

    if not user_text.strip():
        return "❌ Please upload a .txt file or paste some text."

    score = check_plagiarism(user_text)
    verdict = get_verdict(score)

    return render_template('result.html', result=score, verdict=verdict)

if __name__ == '__main__':
    app.run(debug=True)
