"""
Generation
- Groq LLM (supported model)
- Grounded prompt (no hallucination)
"""
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class AnswerGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env")
        self.client = Groq(api_key=api_key)
        # Updated, supported model id
        self.model = "llama-3.3-70b-versatile"
        print(f"✅ Groq LLM initialized: {self.model}")

    def generate_answer(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a helpful assistant answering questions about company policies.\n"
            "RULES:\n"
            "1) Answer ONLY from the provided context.\n"
            "2) If not in context, reply: 'I cannot find this information in the provided documents'.\n"
            "3) Cite sources with [Source X].\n"
            "4) Be concise and accurate. No hallucinations."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer (with [Source X] citations):"

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=500,
                top_p=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ Error generating answer: {e}"
