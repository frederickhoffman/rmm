# Reflective Memory Management (RMM) 🧠

Implementation of the paper **"In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents"**.

This project provides a stateful dialogue agent that proactively manages its long-term memory and adaptively refines its retrieval mechanism through reinforcement learning.

## 📖 Problem Formulation

Building personalized dialogue agents for multi-session interactions presents two critical challenges that RMM addresses:

1.  **Proactive Information Management**: Agents must proactively identify, extract, and store salient information from dialogue sessions, anticipating future retrieval needs before they arise.
2.  **Accurate Information Retrieval**: Agents must accurately retrieve only the relevant past information. Irrelevant context can "distract" Large Language Models (LLMs), leading to degraded response quality and hallucinations.

Existing approaches often suffer from **rigid memory granularity** (e.g., fixed window sizes) and **static retrieval mechanisms** that fail to adapt to evolving user interaction patterns and varied conversational contexts.

## 🏗️ Framework Overview: Reflective Memory Management (RMM)

RMM introduces a dual-reflection architecture that balances comprehensive storage with precise, adaptive retrieval:

### 1. Prospective Reflection (Proactive)
- **Mechanism**: Operates in the "background" to organize dialogue history into topic-based memory representations.
- **Granularity**: Dynamically decomposes interactions across multiple levels—utterances, turns, and sessions—into a personalized memory bank.
- **Process**: Uses LLM-based summarization to extract salient snippets and consolidate them into a coherent long-term knowledge base, minimizing redundancy while maximizing coverage.

### 2. Retrospective Reflection (Retrospective)
- **Mechanism**: Dynamically refines the retrieval process during active dialogue using online feedback signals.
- **Neuronal Re-ranking**: A lightweight re-ranker processes embeddings of the query and candidate memories. It uses the **Gumbel trick** for stochastic sampling to balance exploration and exploitation.
- **Online RL**: The system treats LLM-generated **citations** as a reward signal ($R=1$ if a memory is cited, $R=0$ otherwise). This allows the retriever to iteratively learn which types of memories are most useful for generating accurate, personalized responses.

---

## 🚀 Installation

This project uses `uv` for lightning-fast dependency management.

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd rmm
   ```

2. **Setup environment**:
   ```bash
   uv sync
   cp .env.example .env
   ```
   *Fill in your `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, and `WANDB_API_KEY` in the `.env` file.*

3. **Active virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

---

## 📊 Benchmarking & Reproduction

To reproduce the performance results on **MSC (Multi-Session Chat)** and **LongMemEval**:

1. **Run Evaluation Suite**:
   ```bash
   uv run python src/eval.py
   ```
   This script will:
   - Download the MSC dataset.
   - Run the RMM agent through multiple interactive sessions.
   - Log metrics (Recall@K, Citation Accuracy, BERTScore) to **LangSmith** and **Weights & Biases**.

2. **View Dashboards**:
   - **LangSmith**: Detailed traces of the reflection state machine.
   - **Weights & Biases**: Real-time charts comparing RMM against vanilla retrieval baselines.

---

## �️ Interactive UI (LangSmith / LangGraph Studio)

You can interact with the agent's state machine visually and debug the reflection loops using the LangGraph server:

1. **Start the LangGraph Server**:
   ```bash
   uv run langgraph dev
   ```

2. **Open the UI**:
   - The command will provide a local URL (usually `http://localhost:2024`).
   - Open this URL in your browser to access the **LangGraph Studio**.
   - **Testing Multi-User Memory**: In the "Configuration" or "Input" tab of the Studio, you can set the `user_id` in the input JSON (e.g., `{"messages": [...], "user_id": "user_123"}`).
   - This allows you to verify that the agent retrieves different memories for different users.

3. **Tracing in LangSmith**:
   - All runs from the UI and CLI are automatically traced in your LangSmith project (`rmm-reflection`).
   - Use the traces to inspect the exact input/output of each node (e.g., the specific memories retrieved).

## 💬 CLI Usage

For a quick terminal-based interaction:

```bash
uv run python main.py
```

Observe how the agent "reflects" on the conversation and cites its memories as it learns your preferences over time.

---

## 📁 Repository Structure

- `src/memory_store.py`: Vector Database wrapper (ChromaDB).
- `src/prospective.py`: Logic for summarization and consolidation.
- `src/retrospective.py`: Neuronal re-ranker and RL update logic.
- `src/graph.py`: LangGraph state machine definition.
- `src/eval.py`: Benchmarking and dataset loaders.
- `main.py`: Interactive CLI entry point.

---

## 📜 Citation

If you use this implementation, please cite the original paper:

Tan, Z., Yan, J., Hsu, I.-H., Han, R., Wang, Z., Le, L. T., Song, Y., Chen, Y., Palangi, H., Lee, G., Iyer, A., Chen, T., Liu, H., Lee, C.-Y., & Pfister, T. (2025). In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, (pp. 8416–8439). Association for Computational Linguistics.
