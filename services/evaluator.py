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
            Answer:
            <Direct and concise answer to the question>

            Conditions:
            - <Condition/requirement/exclusion 1>
            - <Condition/requirement/exclusion 2>
            - ...

            Evidence:
            - "<Exact supporting quote from context>"
            - "<Exact supporting quote from context>"

            **5. Insufficient information**:
            If the provided excerpts do not fully answer the question, say:
            "The provided documents do not contain enough information to answer this."

            ---

            ### Examples

            **Example 1**  
            Context:  
            "Section 4: The policy covers surgical treatments for orthopedic conditions, including knee and hip replacement surgeries, provided the insured has completed a 2-year continuous coverage period. Cosmetic procedures are excluded."  

            Question:  
            "Does this policy cover knee surgery?"  

            Expected Answer:  
            Answer: Yes, the policy covers knee surgery.  

            Conditions:  
            - The insured must have completed 2 years of continuous coverage.  
            - Cosmetic procedures are excluded.  

            Evidence:  
            - "The policy covers surgical treatments for orthopedic conditions, including knee and hip replacement surgeries."  
            - "Provided the insured has completed a 2-year continuous coverage period."  

            ---

            **Example 2**  
            Context:  
            "Clause 8.2: Damages resulting from floods are excluded unless the property owner has purchased an additional flood protection rider."  

            Question:  
            "Is flood damage covered?"  

            Expected Answer:  
            Answer: No, flood damage is not covered under the standard policy.  

            Conditions:  
            - Flood damage is only covered if an additional flood protection rider is purchased.  

            Evidence:  
            - "Damages resulting from floods are excluded unless the property owner has purchased an additional flood protection rider."  

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
