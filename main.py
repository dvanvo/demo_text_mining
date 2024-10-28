import streamlit as st
from keybert import KeyBERT
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import os
from collections import defaultdict, Counter
import re
import pandas as pd
import py_vncorenlp  # for VnCoreNLP vietnamese processing
import math
import networkx as nx  # Lib for graph
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from matplotlib import use 

# Streamlit start
st.set_page_config(page_title="Summary & Keywords Extraction", layout="wide")



""" Introduction:

    This is programe demo for text mining.
    
    Project
    -	Objectif: building a module for keywords extraction
    -	Input: individual document
    -	Output: keywords
    -	Implementation plan
        1.	Choosing one graph-based method to implement (TextRank for example).
        2.	Implementing a KeyBERT method (using cosine similarity on BERT embedding method).
        3.	Analyzing the obtained results.

    """
@st.cache_resource  # Ensures the model loads only once
def load_vncorenlp_model():
    vncorenlp_dir = r"D:/CodePy/demo_text_mining/vncorenlp/"
    model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_dir, annotators=["wseg"])
    return model

model = load_vncorenlp_model()
 
# Stopwords Vietnamese
stopwords_file  = r"D:/CodePy/demo_text_mining/vietnamese-stopwords.txt"
 

try:
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        STOPWORDS = file.read().splitlines()
except FileNotFoundError:
    st.error("File stopwords does not exist.")
    STOPWORDS = []

#Read file docx
def read_docx_file(file_path): 
    doc = Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

# Word segment VnCoreNLP
def tokenize_and_filter(text): 
    
    tokens = model.word_segment(text)  # return words tokens list

    filtered_text = []
    for sentence in tokens:
        if isinstance(sentence, str):  # If `sentence` is string
            sentence = sentence.split()  # Segment (split) sentence to tokens

        filtered_sentence = [word for word in sentence if word.lower() not in STOPWORDS] 
        filtered_text.extend(filtered_sentence)  
    
    return filtered_text  # Return list tokens

#Built graph, link tokens together

     
def build_graph(tokens, window_size=2):
    
    graph = nx.Graph()  # init

    # Create edges between words within the context window range
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        for word1, word2 in combinations(window, 2):
            if graph.has_edge(word1, word2):
                graph[word1][word2]['weight'] += 1
            else:
                graph.add_edge(word1, word2, weight=1)

    return graph

def textrank(graph, max_iter=100, damping=0.85):
    
    return nx.pagerank(graph, max_iter=max_iter, alpha=damping)

# Extract keywords textrank:

#     - Extract keywords based on threshold selection strategy.
#     - Mothod: 'relative' or 'mean_std'
   
def extract_keywords_textrank(text, strategy='relative', top_percent=0.1):
    tokens = tokenize_and_filter(text)

    graph = build_graph(tokens)  
    scores = textrank(graph)  

    if strategy == 'relative':
        # Selcct top `top_percent` keywords by TextRank score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_n = int(len(sorted_scores) * top_percent)  # Number of keywords
        filtered_keywords = sorted_scores[:top_n]

    elif strategy == 'mean_std':
        # Average and standard deviation
        mean_score = np.mean(list(scores.values()))
        std_score = np.std(list(scores.values()))

        # Filter words with a score of >= average + 1 standard deviation
        filtered_keywords = {word: score for word, score in scores.items() 
                             if score >= mean_score + std_score}
        # Sort keywords by descending point
        filtered_keywords = sorted(filtered_keywords.items(), key=lambda x: x[1], reverse=True)
 

    return filtered_keywords


    # summarize_text:

    # Summarize the text by selecting the sentences with the most keywords.
    # - text: The text to be summarized.
    # - keywords: List of keywords [(word, weight)].
    # - top_n: Number of sentences you want to keep in the summary.
   

def summarize_text(text, keywords, top_n=3):
    
    # Split sentences with spaCy  
    
    sentences = re.split(r'(?<=[.!?])\s+', text)  

    # Create a weighted dictionary of keywords
    keyword_weights = {kw[0].lower(): kw[1] for kw in keywords}

    # Cal score for each sentence
    sentence_scores = {}
    for sentence in sentences:
        words = sentence.lower().split()
        score = sum(keyword_weights.get(word, 0) for word in words)
        sentence_scores[sentence] = score

    # Select `top_n` the sentence with the highest score
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    summary = " ".join([sent for sent, score in top_sentences])

    return summary

#Graph function with vertices as words and display the PageRank score on each vertex. 
def plot_graph_with_scores(graph, scores):
    
    plt.figure(figsize=(12, 8))  

    pos = nx.spring_layout(graph, seed=42) 

    # Draw edges with weights
    nx.draw_networkx_edges(graph, pos, alpha=0.5)

    # Draw nodes (vertices) with a size that depends on the TextRank point
    node_size = [v * 5000 for v in scores.values()]  
    nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color='lightblue')

    # Draw labels for vertices (keywords) and display TextRank points
    nx.draw_networkx_labels(
        graph, pos, 
        labels={node: f"{node}\n{scores[node]:.4f}" for node in graph.nodes()}, 
        font_size=10, font_color='black'
    )

    plt.title("TextRank Graph - Visualization", fontsize=16)
    plt.axis("off")  
    #plt.show()
    st.pyplot(plt)

 
#graph = build_graph(tokens)   
#scores = textrank(graph) 
#plot_graph_with_scores(graph, scores)


# model_name: 
#         - all-MiniLM-L6-v2:
#             Accuracy: Medium
#             Speed: Fast
#             Size: Small
#             Usage: Real-time extraction, large datasets
#         - distiluse-base-multilingual-cased-v2:
#             Accuracy: Medium
#             Speed:  Fast
#             Size: Small
#             Usage: Multilingual, non-English texts
#         - paraphrase-mpnet-base-v2:
#             Accuracy: High
#             Speed: Moderate
#             Size: Medium
#             Usage: Balanced extraction with good performance
#         - paraphrase-distilroberta-base-v1:
#             Accuracy: High
#             Speed: Fast
#             Size: Medium
#             Usage: Speed-optimized with good accuracy
#         - all-mpnet-base-v2: 
#             Accuracy: Very High
#             Speed: Slow
#             Size: Large
#             Usage: esearch and high-quality extraction
#         - sentence-t5-base:
#             Accuracy: Very High
#             Speed: Slow
#             Size: Large
#             Usage: NLP research and top-tier quality


def get_keyword_model(model_name="all-MiniLM-L6-v2"):
    return KeyBERT(model_name)
 
# init KeyBERT model
#kw_model = get_keyword_model("paraphrase-distilroberta-base-v1")


# Join tokens into cleaned text for keyword extraction
def preprocess_for_keybert(tokens):
    return " ".join(tokens)  # Convert token list to string

# Visualize keywords using WordCloud
def visualize_keywords(keywords):
    word_freq = {kw: score for kw, score in keywords}
    wordcloud = WordCloud(font_path='arial', background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    #plt.show()
    st.pyplot(plt)

def extract_keywords_keybert(model,text,top_percent=0.1):
    
    tokens = tokenize_and_filter(text)
    cleaned_text = preprocess_for_keybert(tokens)

    # Initialize KeyBERT model
    #kw_model = KeyBERT(model='paraphrase-MiniLM-L12-v2')  # Use optimized transformer model
    kw_model = get_keyword_model(model)
    # Calculate the number of keywords to extract (e.g., 10% of tokens)
    keyword_percentage = top_percent  # 10%
    num_keywords = max(1, int(len(tokens) * keyword_percentage))  # Ensure at least 1 keyword
    
    # print("Number of tokens: ",len(tokens))
    # print("Number of keywords ",num_keywords)

    # Extract keywords with MMR for diversity
    keywords_mmr = kw_model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=(1, 2),  # Extract unigrams and bigrams
        stop_words=list(STOPWORDS),
        use_mmr=True,  # Maximal Marginal Relevance for relevance and diversity
        diversity=0.7,  # Adjust diversity between 0 and 1
        top_n=num_keywords  # Extract keywords based on percentage
    )

    return keywords_mmr




# Sidebar - select model
st.sidebar.title("🔍 Choose model")

model_option = st.sidebar.radio("Choose model:", ["TextRank", "KeyBERT"])

top_percent = st.sidebar.slider("% keywords:", 0.05, 0.5, 0.1)
num_sentences = st.sidebar.slider("Number of summary sentences:", 1, 10, 3)


# Main Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Demo Application Extract Keywords And Text Summarize</h1>", unsafe_allow_html=True)

# upload file DOCX
uploaded_file = st.file_uploader("📂 Upload file DOCX", type=["docx"], help="Only support file DOCX.")

 
md_name = ""

if uploaded_file is not None:
    try:
        doc_text = read_docx_file(uploaded_file)

        # show file content
        st.markdown("### 📄 Content:")
        with st.expander("View text content", expanded=True):
            st.write(doc_text)
        # Create column with 2 function buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Keywords extraction"):
                with st.spinner("keywords extraction processing..."):
                    if model_option == "TextRank":
                        keywords = extract_keywords_textrank(doc_text,strategy='relative', top_percent=top_percent)
                        md_name = "TextRank"
                    else:
                        keywords = extract_keywords_keybert("paraphrase-distilroberta-base-v1",doc_text,top_percent=top_percent)
                        md_name = "KeyBERT"

                    keywords_text = "\n".join([f"- **{kw[0]}**: {kw[1]:.4f}" for kw in keywords])
                   
                    st.success("✅ Keywords extraction completed!")   
                    st.markdown(f"**Keywords extraction list, model**: {md_name}\n\n{keywords_text}")                 
                    #st.markdown(f"** {md_name} Keywords extraction:** {', '.join([kw[0] for kw in keywords])}")                    

                    if model_option == "TextRank":
                        # Built and show TextRank graph
                        tokens = tokenize_and_filter(doc_text)
                        graph = build_graph(tokens)
                        scores = textrank(graph)

                        st.markdown("### 📈 Graph TextRank")
                        plot_graph_with_scores(graph, scores)
                        
                  
                    if model_option == "KeyBERT": 
                        visualize_keywords(keywords)
                        
        with col2:
            if st.button("📋 Text Summarize"):
                with st.spinner("Text Summarize processsing..."):
                    if model_option == "TextRank":
                        keywords = extract_keywords_textrank(doc_text,strategy='relative', top_percent= top_percent)
                        md_name = "TextRank"
                    else:
                        keywords = extract_keywords_keybert("paraphrase-distilroberta-base-v1",doc_text,top_percent= top_percent)
                        md_name = "KeyBERT"
                  
                    #keyword_list = [kw[0] for kw in keywords]

                    keywords_text = "\n".join([f"- **{kw[0]}**: {kw[1]:.4f}" for kw in keywords])

                    summary = summarize_text(doc_text, keywords, top_n=num_sentences)
                    st.success("✅ Text Summarize completed!")
 
                    st.markdown(f"**Keywords extraction list, model**: {md_name}\n\n{keywords_text}")     

                    st.markdown(f"### 📑 {md_name} Summary: \n{summary}")
    except Exception as e:
        st.error(f"⚠️ Error file reading: {str(e)}")

else:
    st.info("Upload file DOCX to starting.")

