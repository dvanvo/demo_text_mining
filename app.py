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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

#Ham tach cau
# def preprocess_sentences(text):
#     #sentences = text.split('.')  # Basic sentence split; adjust as needed.
#     sentences = re.split(r'(?<=[.!?])\s+', text)
#     return [sentence.strip() for sentence in sentences if sentence]

  
# #Tinh khoang cach - do tuong tu giua cac cau
# def compute_similarity(sentences):
#     vectorizer = TfidfVectorizer().fit_transform(sentences)
#     similarity_matrix = cosine_similarity(vectorizer)
#     return similarity_matrix

# #Sep hang - tinh diem 
# def rank_sentences(similarity_matrix):
#     nx_graph = nx.from_numpy_array(similarity_matrix)
#     scores = nx.pagerank(nx_graph)
#     return scores

# #Tom tat van ban - theo tỉ % so cau ; mặc định tóm tắt 30%
# def summarize(text, top_percent=0.3):
#     sentences = preprocess_sentences(text)
#     similarity_matrix = compute_similarity(sentences)
#     scores = rank_sentences(similarity_matrix)
    
#     # Dua vao top_percent => tinh ra so cau can lay tom tat
#     top_n = max(1, int(len(sentences) * top_percent))
    
#     ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
#     summary = " ".join([s for _, s in ranked_sentences[:top_n]])
#     return summary 

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

#Tach cau
def preprocess_sentences(text): 
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [sentence.strip() for sentence in sentences if sentence]
  

def compute_word_vectors(tokenized_sentences):
    all_words = [" ".join(sentence) for sentence in tokenized_sentences]
    vectorizer = TfidfVectorizer().fit(all_words)  # Tạo TF-IDF dựa trên tất cả các từ trong văn bản
    
    # Tính vector TF-IDF cho từng từ trong mỗi câu
    word_vectors = []
    for sentence in tokenized_sentences:
        vectors = vectorizer.transform(sentence).toarray()  # Vector hóa từng từ trong câu
        word_vectors.append(vectors)
    
    return word_vectors

# Tính khoảng cách giữa các câu bằng cách lấy trung bình khoảng cách giữa các từ
def compute_sentence_similarity(word_vectors):
    num_sentences = len(word_vectors)
    similarity_matrix = np.zeros((num_sentences, num_sentences))
    
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                # Tính khoảng cách cosine giữa các từ trong hai câu
                word_similarities = cosine_similarity(word_vectors[i], word_vectors[j])
                # Tính khoảng cách giữa hai câu bằng cách lấy trung bình khoảng cách giữa các từ
                similarity_matrix[i][j] = word_similarities.mean()
    
    return similarity_matrix

# def pagerank(graph, damping_factor=0.15, max_iterations=100, tol=1.0e-6):
#     n = graph.shape[0]
#     rank = np.ones(n) / n  # Khởi tạo điểm PageRank ban đầu đều nhau
#     for _ in range(max_iterations):
#         new_rank = np.ones(n) * (damping_factor / n) + (1 - damping_factor) * graph.T.dot(rank / np.maximum(graph.sum(axis=1), 1))
#         if np.linalg.norm(new_rank - rank, ord=1) < tol:  # Kiểm tra độ hội tụ
#             break
#         rank = new_rank
#     return rank

def pagerank(graph, damping_factor=0.15, max_iterations=100, tol=1.0e-6):
    n = graph.shape[0]
    rank = np.ones(n) / n  # Initialize PageRank scores evenly
    degree = graph.sum(axis=1)  # Calculate the degree of each node (sentence)
    
    for _ in range(max_iterations):
        # Calculate new rank
        new_rank = np.ones(n) * (damping_factor / n) + (1 - damping_factor) * graph.T.dot(rank * degree) / np.maximum(degree, 1)
        if np.linalg.norm(new_rank - rank, ord=1) < tol:  # Check for convergence
            break
        rank = new_rank
    return rank

#Xây dựng đồ thị và tính điểm cho các câu dựa trên ma trận khoảng cách
def rank_sentences(similarity_matrix):
    scores = pagerank(similarity_matrix)
    return scores

def summarize(text, top_percent=0.3):
    tokenized_sentences = preprocess_sentences(text)
    word_vectors = compute_word_vectors(tokenized_sentences)
    similarity_matrix = compute_sentence_similarity(word_vectors)
    scores = rank_sentences(similarity_matrix)
    
    # Tính số lượng câu cần lấy dựa trên tỉ lệ top_percent
    top_n = max(1, int(len(tokenized_sentences) * top_percent))
    
    # Xếp hạng câu dựa trên điểm PageRank
    ranked_sentences = sorted(((scores[i], " ".join(sentence)) for i, sentence in enumerate(tokenized_sentences)), reverse=True)
    summary = " ".join([s for _, s in ranked_sentences[:top_n]])
    return summary


#Built graph, link tokens together

#Tao ma tran

#Tinh khoang cach
#  
def compute_cosine_distance(point1, point2):
    norm1 = np.linalg.norm(point1)
    norm2 = np.linalg.norm(point2) 
 
    # Tránh chia cho 0
    if norm1 == 0 or norm2 == 0:
        return 0  # Khoảng cách tối đa nếu một vector là 0

    cosine_similarity = np.dot(point1, point2) / (norm1 * norm2)
    return cosine_similarity  # Cosine distance


     
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


# - Bo sung them nhieu phuong phap summary: su dung Cue Words
# - Phương pháp: Graph based methods
# - Dựa trên kỹ thuật phân cụm
#     + Số lượng câu hoặc % số câu tóm tắt so với văn bản gốc
#     + 
# - Tóm tắt dựa trên nhiều văn bản (tính cùng topic, cùng series)



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

