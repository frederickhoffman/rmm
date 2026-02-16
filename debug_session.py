import json

with open("data/longmemeval_s_cleaned.json", "r") as f:
    data = json.load(f)

for item in data:
    if item["question_id"] == "e47becba":
        print(f"Question: {item['question']}")
        print(f"Answer: {item['answer']}")
        print(f"Answer Session IDs: {item['answer_session_ids']}")
        
        # Find the session content for these IDs
        target_ids = set(item["answer_session_ids"])
        print(f"Looking for sessions: {target_ids}")
        
        found_target = False
        for sess, sid in zip(item["haystack_sessions"], item["haystack_session_ids"]):
            if sid in target_ids:
                found_target = True
                print(f"\n--- MATCHING SESSION {sid} ---")
                full_text = ""
                for m in sess:
                    print(f"{m['role']}: {m['content']}")
                    full_text += m['content']
                
                if "Business" in full_text:
                    print(f"\n[CONFIRMED] 'Business' found in text.")
                    start_idx = full_text.find("Business")
                    print(f"CONTEXT: ...{full_text[start_idx-100:start_idx+100]}...")
                else:
                    print(f"\n[WARNING] 'Business' NOT found in text.")
        break
