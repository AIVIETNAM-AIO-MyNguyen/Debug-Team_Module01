import json

with open("data/processed/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print(f"Total questions: {len(questions)}")
fully_mapped = 0
partially_mapped = 0
unmapped = 0

for q in questions:
    chunk_ids = q["ground_truth_chunk_ids"]
    mapped_strategies = [strat for strat, ids in chunk_ids.items() if len(ids) > 0]
    
    if len(mapped_strategies) == 4:
        fully_mapped += 1
    elif len(mapped_strategies) > 0:
        partially_mapped += 1
    else:
        unmapped += 1

print(f"Fully mapped (all 4 strategies): {fully_mapped}")
print(f"Partially mapped (1-3 strategies): {partially_mapped}")
print(f"Unmapped (0 strategies): {unmapped}")

# Print first 2 fully mapped questions
print("\nExample fully mapped questions:")
count = 0
for q in questions:
    chunk_ids = q["ground_truth_chunk_ids"]
    mapped_strategies = [strat for strat, ids in chunk_ids.items() if len(ids) > 0]
    if len(mapped_strategies) == 4:
        print(f"\nID: {q['id']}")
        print(f"Question: {q['question']}")
        print(f"Ground Truth IDs: {chunk_ids}")
        count += 1
        if count >= 2:
            break
