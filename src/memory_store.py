import os
from typing import List, Optional
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

class MemoryEntry(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)

from langchain_core.documents import Document

class MemoryStore:
    def __init__(self, collection_name: str = "rmm_long_term_memory", persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def add_memories(self, entries: List[MemoryEntry]):
        texts = [e.content for e in entries]
        ids = [e.id for e in entries]
        
        # Ensure 'id' is in metadata for easy retrieval and sanitize for Chroma
        metadatas = []
        for e in entries:
            meta = {**e.metadata, "id": e.id}
            # Chroma doesn't like empty lists in metadata
            for k, v in list(meta.items()):
                if isinstance(v, list) and len(v) == 0:
                    meta[k] = ["NONE"] 
            metadatas.append(meta)
            
        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    def retrieve(self, query: str, k: int = 5, user_id: Optional[str] = None) -> List[MemoryEntry]:
        filter = {"user_id": user_id} if user_id else None
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k, filter=filter)
        memories = []
        for doc, score in results:
            memories.append(MemoryEntry(
                id=doc.metadata.get("id", ""),
                content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": score}
            ))
        return memories

    def get_all_memories(self, user_id: Optional[str] = None) -> List[MemoryEntry]:
        # This is for the consolidator to check similarity
        where = {"user_id": user_id} if user_id else None
        docs = self.vector_store.get(where=where)
        memories = []
        for i in range(len(docs["ids"])):
            memories.append(MemoryEntry(
                id=docs["ids"][i],
                content=docs["documents"][i],
                metadata=docs["metadatas"][i]
            ))
        return memories

    def update_memory(self, memory_id: str, new_content: str, new_metadata: Optional[dict] = None):
        doc = Document(page_content=new_content, metadata=new_metadata or {})
        self.vector_store.update_document(
            document_id=memory_id,
            document=doc
        )
