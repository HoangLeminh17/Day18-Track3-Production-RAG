"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    
    # Split text into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    
    # Encode sentences
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, show_progress_bar=False)
    
    # Cosine similarity helper
    from numpy import dot
    from numpy.linalg import norm
    def cosine_sim(a, b):
        return dot(a, b) / (norm(a) * norm(b) + 1e-8)
    
    # Group sentences by similarity
    chunks = []
    current_group = [sentences[0]]
    
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold:
            # Start new chunk
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])
    
    # Don't forget last group
    if current_group:
        chunks.append(Chunk(
            text=" ".join(current_group),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
        ))
    
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    
    # Step 1: Split text into parents
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parents = []
    children = []
    
    current_parent = ""
    parent_index = 0
    
    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            # Create parent chunk
            pid = f"parent_{parent_index}"
            parents.append(Chunk(
                text=current_parent.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "parent_index": parent_index}
            ))
            
            # Create children from this parent
            parent_text = current_parent.strip()
            child_index = 0
            for i in range(0, len(parent_text), child_size):
                child_text = parent_text[i:i + child_size].strip()
                if child_text:
                    children.append(Chunk(
                        text=child_text,
                        metadata={**metadata, "chunk_type": "child", "child_index": child_index},
                        parent_id=pid
                    ))
                    child_index += 1
            
            current_parent = ""
            parent_index += 1
        
        current_parent += para + "\n\n"
    
    # Don't forget last parent
    if current_parent.strip():
        pid = f"parent_{parent_index}"
        parents.append(Chunk(
            text=current_parent.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "parent_index": parent_index}
        ))
        
        # Create children from last parent
        parent_text = current_parent.strip()
        child_index = 0
        for i in range(0, len(parent_text), child_size):
            child_text = parent_text[i:i + child_size].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child", "child_index": child_index},
                    parent_id=pid
                ))
                child_index += 1
    
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)
    chunks = []
    current_header = ""
    current_content = ""

    def flush_section() -> None:
        if current_header.strip() or current_content.strip():
            section_text = f"{current_header}\n{current_content}".strip()
            if section_text:
                chunks.append(Chunk(
                    text=section_text,
                    metadata={**metadata, "section": current_header.strip(), "strategy": "structure", "chunk_index": len(chunks)}
                ))

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            flush_section()
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    flush_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def summarize(chunks: list[Chunk]) -> dict:
        lengths = [len(chunk.text) for chunk in chunks]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_chunks = []
    semantic_chunks = []
    parent_chunks = []
    child_chunks = []
    structure_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc.get("metadata", {})
        basic_chunks.extend(chunk_basic(text, metadata=metadata))
        semantic_chunks.extend(chunk_semantic(text, metadata=metadata))
        parents, children = chunk_hierarchical(text, metadata=metadata)
        parent_chunks.extend(parents)
        child_chunks.extend(children)
        structure_chunks.extend(chunk_structure_aware(text, metadata=metadata))

    results = {
        "basic": summarize(basic_chunks),
        "semantic": summarize(semantic_chunks),
        "hierarchical": {
            **summarize(child_chunks),
            "parent_chunks": len(parent_chunks),
            "child_chunks": len(child_chunks),
        },
        "structure": summarize(structure_chunks),
    }

    print("Strategy      | Chunks | Avg Len | Min | Max")
    print("-" * 50)
    for name, stats in results.items():
        print(f"{name:<13} | {stats['num_chunks']:>6} | {stats['avg_length']:>7.1f} | {stats['min_length']:>3} | {stats['max_length']:>3}")

    return results
    # 4. Return results dict
    return {}


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
