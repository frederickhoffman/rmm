import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from src.memory_store import MemoryStore, MemoryEntry

class RerankerModel(nn.Module):
    def __init__(self, embedding_dim: int):
        super(RerankerModel, self).__init__()
        # Input is concatenated [query_embedding, memory_embedding]
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, query_emb: torch.Tensor, memory_embs: torch.Tensor) -> torch.Tensor:
        """
        query_emb: (1, dim)
        memory_embs: (K, dim)
        returns scores: (K,)
        """
        K = memory_embs.shape[0]
        query_expanded = query_emb.expand(K, -1)
        combined = torch.cat([query_expanded, memory_embs], dim=-1)
        scores = self.net(combined).squeeze(-1)
        return scores

class RetrospectiveReflection:
    def __init__(self, memory_store: MemoryStore, embedding_dim: int = 1536):
        self.memory_store = memory_store
        self.embeddings = OpenAIEmbeddings()
        self.model = RerankerModel(embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_embeddings(self, texts: List[str]) -> torch.Tensor:
        embs = self.embeddings.embed_documents(texts)
        return torch.tensor(embs, dtype=torch.float32)

    def rerank(self, query: str, memories: List[MemoryEntry]) -> Tuple[List[MemoryEntry], torch.Tensor]:
        if not memories:
            return [], torch.tensor([])

        query_emb = self.get_embeddings([query])
        memory_texts = [m.content for m in memories]
        memory_embs = self.get_embeddings(memory_texts)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(query_emb, memory_embs)
        
        # Sort memories by scores
        indexed_scores = list(enumerate(scores.tolist()))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        sorted_memories = [memories[i] for i, _ in indexed_scores]
        return sorted_memories, scores

    def update_weights(self, query: str, memories: List[MemoryEntry], citations: List[str]):
        """
        Update the re-ranker based on which memories were actually cited.
        Reward signal: 1 if cited, 0 otherwise.
        """
        if not memories:
            return

        query_emb = self.get_embeddings([query])
        memory_texts = [m.content for m in memories]
        memory_embs = self.get_embeddings(memory_texts)

        # Labels: 1 if memory content is cited in the response
        # The prompt uses [Memory i] format where i is the index in the input list
        cited_indices = set()
        for cite in citations:
            match = re.search(r"\[Memory (\d+)\]", cite)
            if match:
                cited_indices.add(int(match.group(1)))

        labels = []
        for i, mem in enumerate(memories):
            # Check if this index was cited
            # Fallback: check if content is in citation (legacy)
            is_cited = i in cited_indices
            labels.append(1.0 if is_cited else 0.0)
        
        target = torch.tensor(labels, dtype=torch.float32)

        self.model.train()
        self.optimizer.zero_grad()
        scores = self.model(query_emb, memory_embs)
        loss = self.criterion(scores, target)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def save_model(self, path: str = "reranker.pth"):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "reranker.pth"):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
