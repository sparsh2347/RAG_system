# services/evaluator.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro"

class Evaluator:
    """
    Combines retrieved context with the user's query,
    and sends it to Gemini LLM for a final, reasoned answer.
    """

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("❌ GEMINI_API_KEY not found in environment")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = GEMINI_MODEL

    def generate_answer(self, query: str, retrieved_chunks: list, max_output_tokens: int = 512):
        """
        Takes the query + retrieved chunks and returns a detailed insurance/legal-focused answer from Gemini.
        """
        # Merge all retrieved chunk content into one context string
        context_text = "\n\n".join(
            [f"Source: {chunk['metadata']} \nContent: {chunk['chunk']}" for chunk in retrieved_chunks]
        )

        prompt = f"""
           You are an expert analyst for insurance and legal documents.
            Your job is to answer the user's question based ONLY on the provided context.
            If the wording of the question is vague, first interpret what the user might mean using domain knowledge,
            then look for relevant coverage, clauses, or conditions in the context.

            You will receive context chunks extracted from insurance or legal documents.
            Each chunk may include **section headings** in the metadata.
            ---

            **1. Understand and interpret the question**:
            - Grasp the intent even if the question is vague or uses different wording.
            - Recognize synonyms, paraphrases, and related concepts common in insurance/legal contexts.
            - Expand the question mentally to consider relevant clauses, exclusions, time limits, and legal jargon.

            **2. Use section headings for clarity**:
            - Treat headings as strong indicators of topic relevance.
            - If a heading closely relates to the question, highlight it in your answer.
            - Prefer information from chunks with highly relevant headings.
            
            **3. Extract comprehensive, relevant information**:
            - Determine whether the policy/document covers the item or situation in question.
            - List **all** conditions, requirements, limitations, waiting periods, or exclusions that apply.
            - If multiple clauses address the same matter, merge them into a single, clear explanation.

            **4. Ensure precision and auditability**:
            - Base answers strictly on the provided context — no assumptions.
            - Quote exact wording from the document to support each claim.
            - Maintain professional and legally sound language.

            **5. Response format**:
            
            Rules:
            1. Answers must be short, precise, and factual — similar to bullet-point policy statements.
            2. No speculation, no extra explanation, no vague language.
            3. Use the same style as official insurance clauses: clear, concise, formal.
            4. Do not use “maybe”, “generally”, or “it appears”.
            5. If the question cannot be answered from the provided context, reply exactly:  
            "The provided documents do not contain enough information to answer this."
            6. Always base your response strictly on the context — no outside knowledge.
            7. Responses must be **single, self-contained statements** that directly answer the question. 

            Example styles:  
            - "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."  
            - "The policy has a waiting period of 36 months from the first inception date for pre-existing diseases."  
            - "Yes, the policy covers maternity expenses subject to a 24-month continuous coverage requirement."

            Output format:
            <One short, precise factual statement>
            ---
            
            Context:
            {context_text}

            Question:
            {query}
            """

        try:
            response = genai.GenerativeModel(self.model).generate_content(
                prompt,
                generation_config={"max_output_tokens": max_output_tokens}
            )
            return response.text
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return None


# if __name__ == "__main__":
#     # Test with mock retrieved chunks
#     mock_chunks = [
#         {"chunk": "AI is being used in healthcare to assist doctors in diagnosis and treatment.", "metadata": "doc1.pdf"},
#         {"chunk": "Machine learning models help predict patient outcomes and optimize hospital workflows.", "metadata": "doc1.pdf"}
#     ]

#     evaluator = Evaluator()
#     answer = evaluator.generate_answer("How is AI used in healthcare?", mock_chunks)
#     print("\n🤖 Final Answer:\n", answer)
