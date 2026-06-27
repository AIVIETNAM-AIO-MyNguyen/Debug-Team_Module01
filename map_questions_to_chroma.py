import os
import json
import pandas as pd
import chromadb
import re

def main():
    print("Connecting to ChromaDB persistent client...")
    client = chromadb.PersistentClient(path="data/processed/embeddings")
    
    print("Loading confluence_questions.parquet...")
    df = pd.read_parquet("data/raw/confluence_questions.parquet")
    
    collections = {
        "fixed_512": client.get_collection("c_512"),
        "fixed_1024": client.get_collection("c_1024"),
        "recursive": client.get_collection("c_rec"),
        "semantic": client.get_collection("c_sem")
    }
    
    mapped_questions = []
    unmapped_count = 0
    
    print("Mapping questions...")
    for idx, row in df.iterrows():
        q_id = row["id"]
        question = row["question"]
        gt_answer = row["ground_truth_answer"]
        
        # Parse ground_truth_contexts
        contexts = row["ground_truth_contexts"]
        if isinstance(contexts, str):
            contexts = json.loads(contexts)
        elif hasattr(contexts, "tolist"):
            contexts = contexts.tolist()
        else:
            contexts = list(contexts)
            
        source_doc_ids = row["source_document_ids"]
        if isinstance(source_doc_ids, str):
            source_doc_ids = json.loads(source_doc_ids)
        elif hasattr(source_doc_ids, "tolist"):
            source_doc_ids = source_doc_ids.tolist()
        else:
            source_doc_ids = list(source_doc_ids)
            
        gt_chunk_ids = {}
        
        # Match for each chunking strategy (collection)
        for strategy, col in collections.items():
            gt_chunk_ids[strategy] = []
            for ctx in contexts:
                # Clean up query text for contains filter
                # Take first 10 words, remove special chars to avoid parse errors
                words = [w for w in re.findall(r'\w+', ctx) if len(w) > 1]
                search_prefix = " ".join(words[:8])
                
                if not search_prefix:
                    continue
                    
                res = col.get(where_document={"$contains": search_prefix})
                if res["ids"]:
                    # Found match(es), record them
                    gt_chunk_ids[strategy].extend(res["ids"])
                else:
                    # Fallback: try search using raw first 40 characters
                    fallback_prefix = ctx[:40].replace('"', '').replace("'", "").strip()
                    if len(fallback_prefix) > 10:
                        res = col.get(where_document={"$contains": fallback_prefix})
                        if res["ids"]:
                            gt_chunk_ids[strategy].extend(res["ids"])
            
            # Deduplicate IDs
            gt_chunk_ids[strategy] = list(set(gt_chunk_ids[strategy]))
            
        # Check if we successfully mapped at least one chunk ID for each strategy
        is_mapped_fully = all(len(gt_chunk_ids[strat]) > 0 for strat in collections)
        if not is_mapped_fully:
            unmapped_count += 1
            
        mapped_questions.append({
            "id": q_id,
            "question": question,
            "ground_truth_answer": gt_answer,
            "ground_truth_contexts": contexts,
            "ground_truth_chunk_ids": gt_chunk_ids,
            "source_document_ids": source_doc_ids
        })
        
    print(f"Mapped {len(mapped_questions)} questions. {unmapped_count} questions had incomplete mappings.")
    
    # Save to data/processed/questions_mapped.json
    output_path = "data/processed/questions_mapped.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapped_questions, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
        
    print(f"Saved mapped questions to {output_path}")
    import sys
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
