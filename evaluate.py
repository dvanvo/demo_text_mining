import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd

class KeywordSummaryEvaluator:
    def __init__(self, sentence_model="paraphrase-mpnet-base-v2"):#all-MiniLM-L6-v2
        self.sentence_model = SentenceTransformer(sentence_model)
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_keywords(self, text: str, textrank_keywords: List[str], 
                         keybert_keywords: List[str]) -> Dict:
        """
        Đánh giá chất lượng từ khóa được trích xuất bởi TextRank và KeyBERT
        """
        # Tính độ đa dạng của từ khóa
        textrank_diversity = self._keyword_diversity(textrank_keywords)
        keybert_diversity = self._keyword_diversity(keybert_keywords)
        
        # Tính độ phủ văn bản
        textrank_coverage = self._keyword_coverage(text, textrank_keywords)
        keybert_coverage = self._keyword_coverage(text, keybert_keywords)
        
        # Tính độ tương đồng ngữ nghĩa giữa từ khóa và văn bản
        textrank_relevance = self._keyword_relevance(text, textrank_keywords)
        keybert_relevance = self._keyword_relevance(text, keybert_keywords)
        
        # So sánh độ tương đồng giữa hai tập từ khóa
        keyword_similarity = self._compare_keyword_sets(textrank_keywords, keybert_keywords)
        
        return {
            'keyword_metrics': {
                'textrank': {
                    'diversity': textrank_diversity,
                    'coverage': textrank_coverage,
                    'relevance': textrank_relevance
                },
                'keybert': {
                    'diversity': keybert_diversity,
                    'coverage': keybert_coverage,
                    'relevance': keybert_relevance
                },
                'similarity_between_methods': keyword_similarity
            }
        }
    
    def evaluate_summaries(self, original_text: str, textrank_summary: str, 
                          keybert_summary: str) -> Dict:
        """
        Đánh giá chất lượng tóm tắt được tạo bởi TextRank và KeyBERT
        """
        # Tính điểm ROUGE
        textrank_rouge = self._calculate_rouge_scores(original_text, textrank_summary)
        keybert_rouge = self._calculate_rouge_scores(original_text, keybert_summary)
        
        # Tính độ tương đồng ngữ nghĩa
        textrank_semantic = self._semantic_similarity(original_text, textrank_summary)
        keybert_semantic = self._semantic_similarity(original_text, keybert_summary)
        
        # Tính độ ngắn gọn
        textrank_conciseness = self._calculate_conciseness(original_text, textrank_summary)
        keybert_conciseness = self._calculate_conciseness(original_text, keybert_summary)
        
        return {
            'summary_metrics': {
                'textrank': {
                    'rouge_scores': textrank_rouge,
                    'semantic_similarity': textrank_semantic,
                    'conciseness': textrank_conciseness
                },
                'keybert': {
                    'rouge_scores': keybert_rouge,
                    'semantic_similarity': keybert_semantic,
                    'conciseness': keybert_conciseness
                }
            }
        }
    
    def _keyword_diversity(self, keywords: List[str]) -> float:
        """Tính độ đa dạng của từ khóa dựa trên số từ unique"""
        if not keywords:
            return 0.0
        return len(set(keywords)) / len(keywords)
    
    def _keyword_coverage(self, text: str, keywords: List[str]) -> float:
        """Tính tỷ lệ từ khóa xuất hiện trong văn bản gốc"""
        text_words = set(text.lower().split())
        keyword_words = set(' '.join(keywords).lower().split())
        if not text_words:
            return 0.0
        return len(keyword_words.intersection(text_words)) / len(text_words)
    
    def _keyword_relevance(self, text: str, keywords: List[str]) -> float:
        """Tính độ liên quan ngữ nghĩa giữa từ khóa và văn bản"""
        if not keywords:
            return 0.0
        text_embedding = self.sentence_model.encode([text])[0]
        keyword_embeddings = self.sentence_model.encode(keywords)
        similarities = [cosine_similarity([text_embedding], [keyword_embedding])[0][0] 
                       for keyword_embedding in keyword_embeddings]
        return np.mean(similarities)
    
    def _compare_keyword_sets(self, keywords1: List[str], keywords2: List[str]) -> float:
        """So sánh độ tương đồng giữa hai tập từ khóa"""
        if not keywords1 or not keywords2:
            return 0.0
        embeddings1 = self.sentence_model.encode(keywords1)
        embeddings2 = self.sentence_model.encode(keywords2)
        
        # Tính ma trận tương đồng
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        return np.mean(similarity_matrix)
    
    def _calculate_rouge_scores(self, reference: str, summary: str) -> Dict:
        """Tính điểm ROUGE cho bản tóm tắt"""
        scores = self.rouge_scorer.score(reference, summary)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Tính độ tương đồng ngữ nghĩa giữa hai đoạn văn bản"""
        embedding1 = self.sentence_model.encode([text1])[0]
        embedding2 = self.sentence_model.encode([text2])[0]
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def _calculate_conciseness(self, original: str, summary: str) -> float:
        """Tính độ ngắn gọn của bản tóm tắt"""
        return 1 - (len(summary.split()) / len(original.split()))
    
    def generate_evaluation_report(self, text: str, 
                               textrank_keywords: List[str], 
                               keybert_keywords: List[str],
                               textrank_summary: str, 
                               keybert_summary: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate detailed evaluation report as DataFrames.
        """
        # Evaluation metrics for keywords and summary
        keyword_eval = self.evaluate_keywords(text, textrank_keywords, keybert_keywords)
        summary_eval = self.evaluate_summaries(text, textrank_summary, keybert_summary)
        
        # Create DataFrame for keyword evaluation
        keyword_data = {
            'Metric': ['Diversity', 'Coverage', 'Relevance'],
            'TextRank': [
                keyword_eval['keyword_metrics']['textrank'].get('diversity', 0.0),
                keyword_eval['keyword_metrics']['textrank'].get('coverage', 0.0),
                keyword_eval['keyword_metrics']['textrank'].get('relevance', 0.0)
            ],
            'KeyBERT': [
                keyword_eval['keyword_metrics']['keybert'].get('diversity', 0.0),
                keyword_eval['keyword_metrics']['keybert'].get('coverage', 0.0),
                keyword_eval['keyword_metrics']['keybert'].get('relevance', 0.0)
            ]
        }
        keyword_df = pd.DataFrame(keyword_data)
        
        # Create DataFrame for summary evaluation
        summary_data = {
            'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Similarity', 'Conciseness'],
            'TextRank': [
                summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rouge1', 0.0),
                summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rouge2', 0.0),
                summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rougeL', 0.0),
                summary_eval['summary_metrics']['textrank'].get('semantic_similarity', 0.0),
                summary_eval['summary_metrics']['textrank'].get('conciseness', 0.0)
            ],
            'KeyBERT': [
                summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rouge1', 0.0),
                summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rouge2', 0.0),
                summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rougeL', 0.0),
                summary_eval['summary_metrics']['keybert'].get('semantic_similarity', 0.0),
                summary_eval['summary_metrics']['keybert'].get('conciseness', 0.0)
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Ensure all relevant columns are converted to numeric, replacing non-numeric with 0.0
        keyword_df[['TextRank', 'KeyBERT']] = keyword_df[['TextRank', 'KeyBERT']].applymap(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)
        summary_df[['TextRank', 'KeyBERT']] = summary_df[['TextRank', 'KeyBERT']].applymap(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0.0)
        
        return keyword_df, summary_df


    # def generate_evaluation_report(self, text: str, 
    #                            textrank_keywords: List[str], 
    #                            keybert_keywords: List[str],
    #                            textrank_summary: str, 
    #                            keybert_summary: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    
    #     # Đánh giá từ khóa và tóm tắt
    #     keyword_eval = self.evaluate_keywords(text, textrank_keywords, keybert_keywords)
    #     summary_eval = self.evaluate_summaries(text, textrank_summary, keybert_summary)
        
    #     # Tạo DataFrame cho đánh giá từ khóa
    #     keyword_data = {
    #         'Metric': ['Diversity', 'Coverage', 'Relevance'],
    #         'TextRank': [
    #             keyword_eval['keyword_metrics']['textrank'].get('diversity', 0.0),
    #             keyword_eval['keyword_metrics']['textrank'].get('coverage', 0.0),
    #             keyword_eval['keyword_metrics']['textrank'].get('relevance', 0.0)
    #         ],
    #         'KeyBERT': [
    #             keyword_eval['keyword_metrics']['keybert'].get('diversity', 0.0),
    #             keyword_eval['keyword_metrics']['keybert'].get('coverage', 0.0),
    #             keyword_eval['keyword_metrics']['keybert'].get('relevance', 0.0)
    #         ]
    #     }
    #     keyword_df = pd.DataFrame(keyword_data)
        
    #     # Tạo DataFrame cho đánh giá tóm tắt
    #     summary_data = {
    #         'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Similarity', 'Conciseness'],
    #         'TextRank': [
    #             summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rouge1', 0.0),
    #             summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rouge2', 0.0),
    #             summary_eval['summary_metrics']['textrank']['rouge_scores'].get('rougeL', 0.0),
    #             summary_eval['summary_metrics']['textrank'].get('semantic_similarity', 0.0),
    #             summary_eval['summary_metrics']['textrank'].get('conciseness', 0.0)
    #         ],
    #         'KeyBERT': [
    #             summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rouge1', 0.0),
    #             summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rouge2', 0.0),
    #             summary_eval['summary_metrics']['keybert']['rouge_scores'].get('rougeL', 0.0),
    #             summary_eval['summary_metrics']['keybert'].get('semantic_similarity', 0.0),
    #             summary_eval['summary_metrics']['keybert'].get('conciseness', 0.0)
    #         ]
    #     }
    #     summary_df = pd.DataFrame(summary_data)
        
    #     # Convert all values to float to ensure numeric formatting
    #     keyword_df[['TextRank', 'KeyBERT']] = keyword_df[['TextRank', 'KeyBERT']].astype(float)
    #     summary_df[['TextRank', 'KeyBERT']] = summary_df[['TextRank', 'KeyBERT']].astype(float)
        
    #     return keyword_df, summary_df

    # def generate_evaluation_report(self, text: str, 
    #                              textrank_keywords: List[str], 
    #                              keybert_keywords: List[str],
    #                              textrank_summary: str, 
    #                              keybert_summary: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Tạo báo cáo đánh giá chi tiết dưới dạng DataFrame
    #     """
    #     # Đánh giá từ khóa và tóm tắt
    #     keyword_eval = self.evaluate_keywords(text, textrank_keywords, keybert_keywords)
    #     summary_eval = self.evaluate_summaries(text, textrank_summary, keybert_summary)
        
    #     # Tạo DataFrame cho đánh giá từ khóa
    #     keyword_data = {
    #         'Metric': ['Diversity', 'Coverage', 'Relevance'],
    #         'TextRank': [
    #             keyword_eval['keyword_metrics']['textrank']['diversity'],
    #             keyword_eval['keyword_metrics']['textrank']['coverage'],
    #             keyword_eval['keyword_metrics']['textrank']['relevance']
    #         ],
    #         'KeyBERT': [
    #             keyword_eval['keyword_metrics']['keybert']['diversity'],
    #             keyword_eval['keyword_metrics']['keybert']['coverage'],
    #             keyword_eval['keyword_metrics']['keybert']['relevance']
    #         ]
    #     }
    #     keyword_df = pd.DataFrame(keyword_data)
        
    #     # Tạo DataFrame cho đánh giá tóm tắt
    #     summary_data = {
    #         'Metric': ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Semantic Similarity', 'Conciseness'],
    #         'TextRank': [
    #             summary_eval['summary_metrics']['textrank']['rouge_scores']['rouge1'],
    #             summary_eval['summary_metrics']['textrank']['rouge_scores']['rouge2'],
    #             summary_eval['summary_metrics']['textrank']['rouge_scores']['rougeL'],
    #             summary_eval['summary_metrics']['textrank']['semantic_similarity'],
    #             summary_eval['summary_metrics']['textrank']['conciseness']
    #         ],
    #         'KeyBERT': [
    #             summary_eval['summary_metrics']['keybert']['rouge_scores']['rouge1'],
    #             summary_eval['summary_metrics']['keybert']['rouge_scores']['rouge2'],
    #             summary_eval['summary_metrics']['keybert']['rouge_scores']['rougeL'],
    #             summary_eval['summary_metrics']['keybert']['semantic_similarity'],
    #             summary_eval['summary_metrics']['keybert']['conciseness']
    #         ]
    #     }
    #     summary_df = pd.DataFrame(summary_data)
        
    #     return keyword_df, summary_df

# def display_evaluation_results(keyword_df: pd.DataFrame, summary_df: pd.DataFrame):
#     """
#     Hiển thị kết quả đánh giá trong Streamlit
#     """
#     st.header("📊 Kết quả đánh giá")
    
#     # Hiển thị đánh giá từ khóa
#     st.subheader("🔑 Đánh giá trích xuất từ khóa")
#     st.dataframe(keyword_df.style.format("{:.4f}"))
    
#     # Tạo biểu đồ so sánh
#     keyword_chart_data = pd.melt(keyword_df, id_vars=['Metric'], var_name='Method', value_name='Score')
#     keyword_chart = alt.Chart(keyword_chart_data).mark_bar().encode(
#         x='Method',
#         y='Score',
#         color='Method',
#         column='Metric'
#     ).properties(width=150)
#     st.altair_chart(keyword_chart)
    
#     # Hiển thị đánh giá tóm tắt
#     st.subheader("📝 Đánh giá tóm tắt văn bản")
#     st.dataframe(summary_df.style.format("{:.4f}"))
    
#     # Tạo biểu đồ so sánh cho tóm tắt
#     summary_chart_data = pd.melt(summary_df, id_vars=['Metric'], var_name='Method', value_name='Score')
#     summary_chart = alt.Chart(summary_chart_data).mark_bar().encode(
#         x='Method',
#         y='Score',
#         color='Method',
#         column='Metric'
#     ).properties(width=100)
#     st.altair_chart(summary_chart)

