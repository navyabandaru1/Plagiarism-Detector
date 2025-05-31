from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_plagiarism(text):
    # Example database (texts to compare with)
    database = [
        "This is a sample text for plagiarism checking.",
        "Flask is a micro web framework written in Python.",
        "Machine learning helps systems improve from experience.",
        "Python is a great programming language for beginners."
    ]
    
    # Add user text to the database list
    texts = database + [text]

    # Convert text to numerical format
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compare last text (user's input) with all other texts
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Find the highest similarity score
    max_score = similarity_scores.max()

    # Convert score to percentage
    percentage = round(max_score * 100, 2)

    return f"⚠️ Similarity Score: {percentage}%"
