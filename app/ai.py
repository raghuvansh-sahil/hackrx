from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AI:
    def __init__(self, gemini_model=None):
        self.model = gemini_model
        self.clauses = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def split_into_clauses(self, text, min_length=100):
        raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [chunk for chunk in raw_chunks if len(chunk) >= min_length]

    def build_index_from_file(self, full_text):
        self.clauses = self.split_into_clauses(full_text, min_length=100)
        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 1), stop_words='english', sublinear_tf=True
        ).fit(self.clauses)
        self.tfidf_matrix = self.vectorizer.transform(self.clauses)

    def semantic_search(self, query, top_k=1):
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({"clause": self.clauses[idx], "distance": 1 - sim_scores[idx]})
        return results

    def answer_query(self, questions, top_k=3):
        answers = []
        for question in questions:
            matches = self.semantic_search(question, top_k=top_k)
            context = "\n\n".join([f"Clause:\n{m['clause']}" for m in matches])
            prompt = f"""
                Role: You are a specialised expert for answering questions about insurance, legal, HR, or compliance documents.

                Goal:
                - Provide a direct and concise answer based only on the provided context.
                - Keep the tone plain, clear, and easy to understand.

                Instructions:
                - Use only the given context clauses.
                - Answer in 1-2 short sentences.
                - Do NOT mention clause numbers or include legal citations.
                - Avoid unnecessary explanations or reasoning unless the question demands it.
                - If the answer is not found in the context, clearly say: "The document does not provide this information."
                - Do not use bullet points, markdown, or special formatting.

                Context:
                {context}

                User Query:
                {question}
            """.strip()

            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            answers.append(answer)
        return answers

    def process_and_answer(self, full_text, questions, top_k=1):
        self.build_index_from_file(full_text)
        return self.answer_query(questions, top_k=top_k)