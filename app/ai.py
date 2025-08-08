import faiss
from sentence_transformers import SentenceTransformer

class AI:
    def __init__(self, gemini_model, embed_model_name="all-MiniLM-L6-v2"):
        self.model = gemini_model
        self.embedder = SentenceTransformer(embed_model_name)
        self.clauses = []
        self.embeddings = None
        self.index = None

    def split_into_clauses(self, text, min_length):
        raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
        return [chunk for chunk in raw_chunks if len(chunk) >= min_length]
    
    def build_index_from_file(self, full_text):
        self.clauses = self.split_into_clauses(full_text, min_length=512)
        self.embeddings = self.embedder.encode(self.clauses, batch_size=1024, show_progress_bar=True, convert_to_numpy=True)
        self.embeddings = self.embeddings.astype('float32')

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def semantic_search(self, query, top_k=3):
        query_vec = self.embedder.encode([query], convert_to_numpy=True).astype('float32')
        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.clauses):
                results.append({"clause": self.clauses[idx], "distance": float(dist)})
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
            """


            response = self.model.generate_content(prompt)
            answer = response.text.strip()

            closest_distance = matches[0]["distance"] if matches else 0.0
            confidence = max(0.0, 1.0 - closest_distance / 10.0)

            answers.append(answer)

        return answers
        

    def process_and_answer(self, full_text, questions, top_k=3):
        self.build_index_from_file(full_text)
        answers = self.answer_query(questions, top_k)
        return answers