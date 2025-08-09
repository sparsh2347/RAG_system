import re
import uuid
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
nltk.download("punkt")

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


CHUNK_MAX_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 80
SEMANTIC_SIM_THRESHOLD = 0.68
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_CHUNK_TOKENS = 50


def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def paragraph_split(text: str) -> List[str]:
    """Split into paragraphs by blank lines or section markers."""
    return [p.strip() for p in re.split(r"\n\s*\n|(?=Section\s+\d+)", text) if p.strip()]


def sentence_split(text: str) -> List[str]:
    return sent_tokenize(text)


def tokens_count(text: str) -> int:
    if not text:
        return 0
    if _ENC:
        return len(_ENC.encode(text))
    return len(text.split())


def detect_heading(line: str) -> bool:
    """
    Detect if a line is likely a heading.
    - Short lines with Title Case or ALL CAPS
    - Starts with 'Section', 'Article', numbers, or legal terms
    """
    line_clean = line.strip()
    if len(line_clean.split()) > 12:  # too long to be a heading
        return False

    heading_patterns = [
        r"^(Section|Article)\s+\d+(\.\d+)*",
        r"^\d+(\.\d+)*\s+[A-Z]",
        r"^[A-Z][A-Z\s\-&,]{3,}$",  # ALL CAPS
        r"^(Coverage|Exclusions|Definitions|Eligibility|Benefits|Conditions|Liability)",
    ]
    return any(re.match(p, line_clean) for p in heading_patterns)


class HybridChunker:
    def __init__(self,
                 embedding_model_name: str = EMBEDDING_MODEL,
                 sim_threshold: float = SEMANTIC_SIM_THRESHOLD,
                 chunk_max_tokens: int = CHUNK_MAX_TOKENS,
                 chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS):
        self.model = SentenceTransformer(embedding_model_name)
        self.sim_threshold = sim_threshold
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def _adaptive_threshold(self, sentences: List[str]) -> float:
        avg_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        if avg_len > 20:
            return max(0.63, self.sim_threshold - 0.05)
        return self.sim_threshold

    def _semantic_group_sentences(self, sentences: List[str]) -> List[List[str]]:
        if not sentences:
            return []
        threshold = self._adaptive_threshold(sentences)
        embeddings = self.model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

        groups, group_embeddings, group_sentences = [], [], []
        group_sentences.append(sentences[0])
        group_embeddings.append(np.array(embeddings[0].cpu()))
        group_mean = np.mean(group_embeddings, axis=0)

        for i in range(1, len(sentences)):
            emb = np.array(embeddings[i].cpu())
            sim = util.cos_sim(group_mean, emb).item()
            if sim >= threshold:
                group_sentences.append(sentences[i])
                group_embeddings.append(emb)
                group_mean = np.mean(group_embeddings, axis=0)
            else:
                groups.append(group_sentences)
                group_sentences = [sentences[i]]
                group_embeddings = [emb]
                group_mean = np.mean(group_embeddings, axis=0)

        if group_sentences:
            groups.append(group_sentences)

        return groups

    def _sliding_window_over_groups(self, groups: List[List[str]], headings: List[str]) -> List[Dict]:
        sent_texts, sent_tokens, group_index_of_sentence = [], [], []
        for gi, group in enumerate(groups):
            for s in group:
                sent_texts.append(s)
                sent_tokens.append(tokens_count(s))
                group_index_of_sentence.append(gi)

        chunks, n, start_idx, chunk_idx = [], len(sent_texts), 0, 0

        while start_idx < n:
            total_tokens, end_idx = 0, start_idx
            while end_idx < n and (total_tokens + sent_tokens[end_idx]) <= self.chunk_max_tokens:
                total_tokens += sent_tokens[end_idx]
                end_idx += 1

            if end_idx == start_idx:
                end_idx = min(start_idx + 1, n)
                total_tokens = sum(sent_tokens[start_idx:end_idx])

            chunk_text = " ".join(sent_texts[start_idx:end_idx])
            chunk_groups = sorted(set(group_index_of_sentence[start_idx:end_idx]))

            chunks.append({
                "id": str(uuid.uuid4()),
                "index": chunk_idx,
                "text": chunk_text,
                "metadata": {
                    "start_sentence_idx": start_idx,
                    "end_sentence_idx": end_idx - 1,
                    "token_count": total_tokens,
                    "char_count": len(chunk_text),
                    "group_indices": chunk_groups,
                    "headings": headings.copy()  # store detected headings
                }
            })
            chunk_idx += 1

            if end_idx >= n:
                break

            desired_overlap = min(self.chunk_overlap_tokens, total_tokens)
            overlap_tokens_acc, back_idx = 0, end_idx - 1
            while back_idx >= start_idx and overlap_tokens_acc < desired_overlap:
                overlap_tokens_acc += sent_tokens[back_idx]
                back_idx -= 1
            start_idx = max(start_idx + 1, back_idx + 1)

        if len(chunks) > 1 and chunks[-1]["metadata"]["token_count"] < MIN_CHUNK_TOKENS:
            chunks[-2]["text"] += " " + chunks[-1]["text"]
            chunks[-2]["metadata"]["token_count"] += chunks[-1]["metadata"]["token_count"]
            chunks[-2]["metadata"]["char_count"] = len(chunks[-2]["text"])
            chunks.pop()

        return chunks

    def chunk(self, text: str, source: Optional[str] = None, page_number: Optional[int] = None) -> List[Dict]:
        print("-------Chunking Started!!------")
        cleaned = clean_text(text)
        paragraphs = paragraph_split(cleaned)
        final_chunks, current_headings = [], []

        for section_id, para in enumerate(paragraphs):
            lines = para.split("\n")
            for line in lines:
                if detect_heading(line):
                    current_headings.append(line.strip())

            sentences = sentence_split(para)
            if not sentences:
                continue

            groups = self._semantic_group_sentences(sentences)
            chunks = self._sliding_window_over_groups(groups, current_headings)

            for c in chunks:
                c["metadata"]["source"] = source or "unknown"
                c["metadata"]["section_id"] = section_id
                if page_number is not None:
                    c["metadata"]["page_number"] = page_number
            final_chunks.extend(chunks)
        print("-------Chunking Done!!------")
        print(type(final_chunks))
        return final_chunks

# -------------------------
# Example usage / test
# -------------------------
# if __name__ == "__main__":
#     sample_text = """
#     Section 1: Coverage
#     This policy covers accidental and natural causes. Knee surgery is included under the orthopedic procedures section.
#     The insured must serve a waiting period of ninety days.
    
#     Section 2: Exclusions
#     Cosmetic surgery and elective procedures are excluded.
#     """

#     chunker = HybridChunker()
#     chunks = chunker.chunk(sample_text, source="sample.pdf")
#     for ch in chunks:
#         print("CHUNK INDEX:", ch["index"])
#         print("TOKENS:", ch["metadata"]["token_count"], "CHARS:", ch["metadata"]["char_count"])
#         print("GROUPS:", ch["metadata"]["group_indices"])
#         print(ch["text"][:400])
#         print("-" * 40)
