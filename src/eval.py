import os
import asyncio
import wandb
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from src.memory_store import MemoryStore, MemoryEntry
from src.prospective import ProspectiveReflection
from src.retrospective import RetrospectiveReflection
from src.graph import RMMGraph
from typing import List, Dict, Any

class RMMEval:
    def __init__(self, project_name: str = "rmm-reflection"):
        self.project_name = project_name
        self.client = Client()
        
        # Initialize RMM components
        self.memory_store = MemoryStore(collection_name="rmm_eval")
        # Use gpt-4o for reliability, with high retries
        self.prospective = ProspectiveReflection(self.memory_store, model_name="gpt-4o")
        self.prospective.llm = ChatOpenAI(model="gpt-4o", max_retries=20)
        self.retrospective = RetrospectiveReflection(self.memory_store)
        self.rmm_agent = RMMGraph(self.memory_store, self.prospective, self.retrospective)

    async def run_msc_eval(self, num_samples: int = 5):
        """
        Evaluate on a subset of the MSC (Multi-Session Chat) dataset.
        """
        wandb.init(project=self.project_name, name="msc-repro-run")
        
        # We'll use a smaller, easier to access dataset or simulate MSC for this validation
        # facebook/msc is a bit heavy. For reproduction, we simulate 3 sessions for each sample.
        samples = [
            {
                "previous_sessions": [
                    "User: I love Italian food particularly carbonara.",
                    "User: I'm planning a trip to Rome next month."
                ],
                "query": "Do you remember which city I'm visiting and what my favorite food is?",
                "expected_entities": ["Rome", "carbonara", "Italian"]
            },
            {
                "previous_sessions": [
                    "User: My dog's name is Buddy. He is a Golden Retriever.",
                    "User: I work as a software engineer at a startup."
                ],
                "query": "What do I do for a living and what's my dog's breed?",
                "expected_entities": ["software engineer", "Golden Retriever", "Buddy"]
            }
        ]
        
        results_table = wandb.Table(columns=["Query", "Retrieved", "Citations", "Accuracy", "Recall@K"])
        
        total_recall = 0
        total_acc = 0
        
        for i, sample in enumerate(samples):
            print(f"Processing sample {i+1}...")
            
            # 1. Prospective Reflection (Fill Memory)
            history = [HumanMessage(content=s) for s in sample["previous_sessions"]]
            summaries = await self.prospective.summarize_dialogue(history)
            await self.prospective.consolidate(summaries, user_id=f"eval_user_{i}")
            
            # 2. Agent Execution
            inputs = {"messages": [HumanMessage(content=sample["query"])], "user_id": f"eval_user_{i}"}
            result = await self.rmm_agent.graph.ainvoke(inputs)
            
            response = result["response"]
            memories = [m.content for m in result["memories"]]
            citations = result["citations"]
            
            # 3. Compute Metrics
            # Recall@K: Did the retrieved memories contain the expected entities?
            found_entities = [e for e in sample["expected_entities"] if any(e.lower() in m.lower() for m in memories)]
            recall = len(found_entities) / len(sample["expected_entities"])
            
            # Accuracy: Did the response actually use the correct info?
            correct_in_response = [e for e in sample["expected_entities"] if e.lower() in response.lower()]
            acc = len(correct_in_response) / len(sample["expected_entities"])
            
            total_recall += recall
            total_acc += acc
            
            results_table.add_data(sample["query"], str(memories), str(citations), acc, recall)
            
            wandb.log({
                "sample_recall": recall,
                "sample_acc": acc,
                "loss": result.get("loss", 0)
            })
            
        avg_recall = total_recall / len(samples)
        avg_acc = total_acc / len(samples)
        
        wandb.log({
            "avg_recall": avg_recall,
            "avg_acc": avg_acc,
            "results_table": results_table
        })
        
        print(f"\nEvaluation Complete!")
        print(f"Avg Recall: {avg_recall:.2%}")
        print(f"Avg Accuracy: {avg_acc:.2%}")
        
    async def run_longmem_eval(self, num_samples: int = 5):
        """
        Evaluate on LongMemEval-inspired samples focusing on long-distance recall.
        These samples test if the agent can find specific 'needles' in session history
        that occurred several sessions ago.
        """
        wandb.init(project=self.project_name, name="longmem-eval-run")
        
        samples = [
            {
                "id": "longmem_1",
                "sessions": [
                    "User: I'm allergic to peanuts. Please remember that.",
                    "User: I want to learn more about quantum physics.",
                    "User: I'm planning to move to Japan in two years.",
                    "User: I just finished reading 'The Great Gatsby'."
                ],
                "query": "Can you remind me of any allergies I have and which country I'm moving to?",
                "expected": ["peanuts", "Japan"]
            },
            {
                "id": "longmem_2",
                "sessions": [
                    "User: My project deadline is December 15th.",
                    "User: I prefer using VS Code over Vim.",
                    "User: I started learning Mandarin Chinese last week.",
                    "User: I'm a big fan of jazz music, especially Miles Davis."
                ],
                "query": "What language did I start learning and when is my project due?",
                "expected": ["Mandarin", "December 15th"]
            }
        ]
        
        results_table = wandb.Table(columns=["ID", "Query", "Recall@K", "Accuracy"])
        total_recall = 0
        total_acc = 0
        
        for sample in samples:
            print(f"Testing {sample['id']}...")
            uid = f"longmem_user_{sample['id']}"
            
            # Prospective Reflection for each session
            for sess in sample["sessions"]:
                summaries = await self.prospective.summarize_dialogue([HumanMessage(content=sess)])
                await self.prospective.consolidate(summaries, user_id=uid)
            
            # Retrieve and Respond
            inputs = {"messages": [HumanMessage(content=sample["query"])], "user_id": uid}
            result = await self.rmm_agent.graph.ainvoke(inputs)
            
            # Metrics
            memories = [m.content for m in result["memories"]]
            response = result["response"]
            
            recall = sum(1 for e in sample["expected"] if any(e.lower() in m.lower() for m in memories)) / len(sample["expected"])
            acc = sum(1 for e in sample["expected"] if e.lower() in response.lower()) / len(sample["expected"])
            
            total_recall += recall
            total_acc += acc
            results_table.add_data(sample["id"], sample["query"], recall, acc)
            
        avg_recall = total_recall / len(samples)
        avg_acc = total_acc / len(samples)
        
        wandb.log({
            "longmem_avg_recall": avg_recall,
            "longmem_avg_acc": avg_acc,
            "longmem_results": results_table
        })
        
        print(f"\nLongMemEval Complete!")
        print(f"Avg Recall: {avg_recall:.2%}")
        print(f"Avg Accuracy: {avg_acc:.2%}")
        
        wandb.finish()

    async def run_longmem_benchmark(self, num_samples: int = 10):
        """
        Rigorous benchmark on the actual LongMemEval S-Cleaned dataset.
        """
        file_path = "data/longmemeval_s_cleaned.json"
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return

        print(f"Loading LongMemEval from {file_path} using streaming parser...")
        import ijson
        
        wandb.init(project=self.project_name, name="longmem-rigorous-benchmark")
        
        results_table = wandb.Table(columns=["Question ID", "Question", "Expected Answer", "Agent Response", "Recall@K", "Correctness"])
        
        # We'll collect a pool of samples to choose from, or just take the first N
        # For a truly random sample without loading all, we take the first N*10 and pick N.
        pool = []
        with open(file_path, 'r') as f:
            parser = ijson.items(f, 'item')
            for i, item in enumerate(parser):
                pool.append(item)
                if i >= num_samples * 2: # Keep pool small
                    break
        
        import random
        random.seed(42)
        samples = random.sample(pool, min(num_samples, len(pool)))
        
        total_recall = 0
        total_correct = 0

        for i, sample in enumerate(samples):
            import uuid
            run_id = str(uuid.uuid4())[:8]
            print(f"[{i+1}/{num_samples}] Evaluating Q: {sample['question_id']} (Run: {run_id})...")
            uid = f"longmem_{sample['question_id']}_{run_id}"
            
            # 1. Clear previous memory for this user and fill with haystack
            # (In a real benchmark, we'd reuse the vector store but here we isolate users)
            sessions = sample.get("haystack_sessions", [])
            session_ids = sample.get("haystack_session_ids", [])
            
            # Parallel processing of sessions
            # Reduced concurrency to respectful limit for gpt-4o (30k TPM)
            semaphore = asyncio.Semaphore(3)
            
            async def process_session(sess, sid):
                msgs = []
                for m in sess:
                    if m["role"] == "user":
                        msgs.append(HumanMessage(content=m["content"]))
                    else:
                        msgs.append(AIMessage(content=m["content"]))
                
                # Retry logic for summarization
                for attempt in range(8):
                    try:
                        async with semaphore:
                            summaries = await self.prospective.summarize_dialogue(msgs)
                        break
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            wait_time = (2 ** attempt) + 1
                            print(f"    [WAIT] Rate limit hit. Waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"    [WARN] Summarization failed for session {sid}: {e}")
                            return []
                else:
                    print(f"    [ERR] Summarization failed after retries for session {sid}")
                    return []
                
                entries = []
                for s in summaries:
                    entries.append(MemoryEntry(
                        id=s.id,
                        content=f"Topic: {s.topic}\nSummary: {s.summary}",
                        metadata={
                            "topic": s.topic,
                            "session_id": sid,
                            "user_id": uid
                        }
                    ))
                return entries

            print(f"  Summarizing {len(sessions)} sessions in parallel...")
            tasks = [process_session(sess, sid) for sess, sid in zip(sessions, session_ids)]
            results = await asyncio.gather(*tasks)
            
            all_entries = [entry for sublist in results for entry in sublist]
            if all_entries:
                self.memory_store.add_memories(all_entries)
            print(f"  Added {len(all_entries)} memory entries.")
            
            # 2. Agent Execution
            print(f"  Querying agent...")
            inputs = {"messages": [HumanMessage(content=sample["question"])], "user_id": uid}
            
            # Retry loop for agent execution
            for attempt in range(8):
                try:
                    result = await self.rmm_agent.graph.ainvoke(inputs)
                    break
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        wait_time = (2 ** attempt) + 2
                        print(f"    [WAIT] Agent execution rate limited. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    [ERR] Agent execution failed: {e}")
                        # Skip this sample if agent fails
                        result = {"response": "Error", "memories": []}
                        break
            else:
                 print(f"    [ERR] Agent execution failed after retries.")
                 result = {"response": "Error", "memories": []}
            
            # 3. Compute Rigorous Recall@K
            # Recall = (Were any of the answer_session_ids in the retrieved memories?)
            target_sids = set(sample["answer_session_ids"])
            retrieved_sids = set(m.metadata.get("session_id") for m in result["memories"])
            
            # If any of the target SIDs is in the retrieved SIDs, we count it as a recall success for that question
            recall = 1.0 if (target_sids & retrieved_sids) else 0.0
            
            # 4. Correctness check (LLM-based)
            judge_prompt = f"""Compare the agent's response to the expected answer.
Expected: {sample['answer']}
Agent: {result['response']}
Question: {sample['question']}

Is the agent's response factually correct based on the expected answer? 
Output only 'YES' or 'NO'."""
            judge_llm = ChatOpenAI(model="gpt-4o-mini") # Use a cheaper model for judging if many samples
            judge_res = await judge_llm.ainvoke(judge_prompt)
            is_correct = 1.0 if "YES" in judge_res.content.upper() else 0.0
            
            if is_correct == 0.0:
                print(f"  [MISMATCH] Expected: {sample['answer']}")
                print(f"             Agent:    {result['response']}")
                print(f"             Recall:   {recall}")
                print(f"             Memories: {[m.content for m in result['memories']]}")
            
            total_recall += recall
            total_correct += is_correct
            
            results_table.add_data(
                sample['question_id'], 
                sample['question'], 
                sample['answer'], 
                result['response'], 
                recall, 
                is_correct
            )
            
            wandb.log({
                "question_recall": recall,
                "question_correctness": is_correct
            })
            
        avg_recall = total_recall / num_samples
        avg_correct = total_correct / num_samples
        
        wandb.log({
            "avg_recall_rigorous": avg_recall,
            "avg_correctness_rigorous": avg_correct,
            "results_table": results_table
        })
        
        print(f"\nRigorous Benchmark Complete!")
        print(f"Avg Recall: {avg_recall:.2%}")
        print(f"Avg Correctness: {avg_correct:.2%}")
        
        wandb.finish()

if __name__ == "__main__":
    import json
    from langchain_core.messages import AIMessage
    eval_tool = RMMEval()
    asyncio.run(eval_tool.run_longmem_benchmark(num_samples=50))
