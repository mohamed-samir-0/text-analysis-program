import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from rouge import Rouge
import warnings
from nltk.corpus import stopwords
import nltk
stop_words = set(stopwords.words("english"))

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Suppress warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('AmazonReview.csv').dropna()
data.loc[data['Sentiment'] <= 3, 'Sentiment'] = 0
data.loc[data['Sentiment'] > 3, 'Sentiment'] = 1

# Load the sentiment analysis model
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review']).toarray()
y = data['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)

naive_bayes = MultinomialNB()
naive_bayes.fit(x_train, y_train)

# Load the named entity recognition (NER) model using spaCy
import spacy
nlp = spacy.load("en_core_web_sm")

# Function to clean review text
def clean_review(review): 
    stp_words = set(stopwords.words('english'))
    return " ".join(word for word in review.split() if word not in stp_words)

# Function to calculate similarity between sentences using cosine similarity
def sentence_similarity(sent1, sent2):
    vector1 = [word.lower() for word in sent1]
    vector2 = [word.lower() for word in sent2]
    
    all_words = list(set(vector1 + vector2))
    
    vector1_dist = [vector1.count(word) for word in all_words]
    vector2_dist = [vector2.count(word) for word in all_words]
    
    return 1 - cosine_distance(vector1_dist, vector2_dist)

# Function to create similarity matrix between sentences
def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 != idx2:
                similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    
    return similarity_matrix

# Function to generate summary using TextRank algorithm
def textrank_summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
    filtered_sentences = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences]
    similarity_matrix = build_similarity_matrix(filtered_sentences)
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)
    
    summary = ""
    sentiment_words = set()
    for score, sentence in ranked_sentences[:num_sentences]:
        summary += sentence + " "
        # Determine sentiment words in the sentence
        words = sentence.split()
        for word in words:
            if word.lower() not in stop_words:
                sentiment_words.add(word)
    
    return summary.strip(), list(sentiment_words)

# Function to calculate ROUGE scores
def calculate_rouge(reference, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, reference)
    return scores

# Function to translate text to Arabic
def translate_to_arabic(text):
    from googletrans import Translator
    # Create translator object
    translator = Translator()
    # Translate text to Arabic
    translated_text = translator.translate(text, src='en', dest='ar').text
    return translated_text

# Function to process the input document
def process_document():
    input_text = text_entry.get("1.0", tk.END).strip()
    
    # Sentiment Analysis
    preprocessed_text = clean_review(input_text)
    input_vector = cv.transform([preprocessed_text]).toarray()
    sentiment_label_lr = logistic_regression.predict(input_vector)[0]
    sentiment_label_nb = naive_bayes.predict(input_vector)[0]
    
    if sentiment_label_lr == 1:
        sentiment_lr = "positive"
    elif sentiment_label_lr == 0:
        sentiment_lr = "negative"
    else:
        sentiment_lr = "neutral"
    
    if sentiment_label_nb == 1:
        sentiment_nb = "positive"
    elif sentiment_label_nb == 0:
        sentiment_nb = "negative"
    else:
        sentiment_nb = "neutral"
    
    # Named Entity Recognition using spaCy
    doc = nlp(input_text)
    named_entities = [ent.text for ent in doc.ents]
    
    # Text Summarization
    summary_en, sentiment_words = textrank_summarize(input_text, num_sentences=2)
    
    # Translation to Arabic
    summary_ar = translate_to_arabic(summary_en)
    
    # Update output text
    output_text.delete("1.0", tk.END)
    output_text.tag_configure("title", font=("Helvetica", 12, "bold"), foreground="blue")
    output_text.insert("1.0", f"Sentiment (Logistic Regression): {sentiment_lr}\n", "title")
    output_text.insert(tk.END, f"Sentiment (Naive Bayes): {sentiment_nb}\n", "title")
    output_text.insert(tk.END, "Named Entities:\n", "title")
    output_text.insert(tk.END, f"{', '.join(named_entities)}\n")
    output_text.insert(tk.END, "Summary (English):\n", "title")
    output_text.insert(tk.END, f"{summary_en}\n")
    output_text.insert(tk.END, "Sentiment Words in Summary:\n", "title")
    output_text.insert(tk.END, f"{', '.join(sentiment_words)}\n")
    output_text.insert(tk.END, "Summary (Arabic):\n", "title")
    output_text.insert(tk.END, f"{summary_ar}\n")

# Create the GUI
root = tk.Tk()
root.title("Document Analysis")

style = ttk.Style()
style.theme_use('clam')
style.configure("Background.TFrame", background="#24788F")

main_frame = ttk.Frame(root, padding=(10, 10, 10, 10), style="Background.TFrame")
main_frame.grid(row=0, column=0, sticky="nsew")

input_label = ttk.Label(main_frame, text="Input Document:")
input_label.grid(row=0, column=0, pady=5, padx=5, sticky="w")

text_entry = ScrolledText(main_frame, height=10, width=50)
text_entry.grid(row=1, column=0, pady=5, padx=5, sticky="nsew")

process_button = ttk.Button(main_frame, text="Process Document", command=process_document)
process_button.grid(row=2, column=0, pady=5, padx=5, sticky="nsew")

output_label = ttk.Label(main_frame, text="Output:")
output_label.grid(row=3, column=0, pady=5, padx=5, sticky="w")

output_text = ScrolledText(main_frame, height=20, width=70)
output_text.grid(row=4, column=0, pady=5, padx=5, sticky="nsew")

main_frame.grid_rowconfigure(4, weight=1)
main_frame.grid_columnconfigure(0, weight=1)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

root.mainloop()