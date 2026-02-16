from typing import List, Dict, Any, Optional
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from src.memory_store import MemoryStore, MemoryEntry

class ReflectionSummary(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    summary: str
    entities: List[str]
    sentiment: str

class ExtractedMemory(BaseModel):
    summary: str
    reference: List[int]

class ExtractionOutput(BaseModel):
    extracted_memories: List[ExtractedMemory]

class ProspectiveReflection:
    def __init__(self, memory_store: MemoryStore, model_name: str = "gpt-4o"):
        self.memory_store = memory_store
        self.llm = ChatOpenAI(model=model_name)
        # Load official prompt (hardcoded here for reliability)
        self.summary_prompt = ChatPromptTemplate.from_template("""You are a highly capable data processing agent. Your task is to analyze historical dialogue transcripts and extract structured information for a research database.

CRITICAL: Preservation of QUALITATIVE and QUANTITATIVE details is paramount.
You MUST NOT generalize or skip the following:
1. NUMERICAL VALUES: Exact amounts, prices, frequencies, and counts.
2. PERCENTAGES: Discount rates, probabilities, and shares.
3. DURATIONS/TIMES: Length of events (e.g., "0.5 hours"), specific dates, and times of day.
4. NAMED ENTITIES: Titles of plays, books, movies, specific places, and proper names.

INCORRECT: "User attended a play" or "User got a discount".
CORRECT: "User attended 'The Glass Menagerie'" or "User received a 10% discount".

Analyze the dialogue below and extract all salient entities and summaries.
Task Description: Given a session of dialogue between SPEAKER_1 and SPEAKER_2, extract the personal summaries of SPEAKER_2, with references to the corresponding turn IDs. Ensure the output adheres to the following rules:
    • Output results in JSON format. The top-level key is "extracted_memories". The value should be a list of dictionaries, where each dictionary has the keys "summary" and "reference":
           – summary: A concise personal summary, which captures relevant information about SPEAKER_2’s experiences, preferences, and background, across multiple turns.
           – reference: A list of references, each in the format of [turn_id] indicating where the information appears.
    • If no personal summary can be extracted, return NO_TRAIT.

Example:
INPUT:
    • Turn 0:
           – SPEAKER_1: Did you manage to go out on a run today?
           – SPEAKER_2: Yes, I actually was able to. I am considering joining the local gym. Do you prefer going to the gym?
    • Turn 1:
           – SPEAKER_1: I do actually. I like the controlled environment. I don't want to have to depend on the weather considering where I live.
           – SPEAKER_2: That's why I am thinking about it. I hate to have to run when it's raining, and I feel like it rains here all the time.
OUTPUT:
{{
    "extracted_memories": [
         {{
             "summary": "SPEAKER_2 is considering joining a local gym due to frequent rain affecting outdoor runs.",
             "reference": [0, 1]
         }}
    ]
}}

Task: Follow the JSON format demonstrated in the example above and extract the personal summaries for SPEAKER_2 from the following dialogue session.
Input:
{dialogue}
Output:""")

    def _format_dialogue(self, messages: List[BaseMessage]) -> str:
        """Formats messages into Turn-based SPEAKER_1/SPEAKER_2 format."""
        formatted_turns = []
        # Group messages into pairs if possible, but handle odd numbers
        # Assuming typical User/AI alternation
        
        # We need to map roles. 
        # SPEAKER_1 = Agent (Assistant)
        # SPEAKER_2 = User
        
        # If we only have 2 messages (one turn), it's Turn 0.
        # But messages might not start perfectly.
        
        current_turn_msgs = []
        turn_idx = 0
        
        for i in range(0, len(messages), 2):
            chunk = messages[i:i+2]
            turn_str = f"    • Turn {turn_idx}:\n"
            
            for m in chunk:
                role_label = "SPEAKER_2" if isinstance(m, HumanMessage) else "SPEAKER_1"
                turn_str += f"           – {role_label}: {m.content}\n"
            
            formatted_turns.append(turn_str)
            turn_idx += 1
            
        return "".join(formatted_turns)

    async def summarize_dialogue(self, messages: List[BaseMessage]) -> List[ReflectionSummary]:
        dialogue_str = self._format_dialogue(messages)
        
        structured_llm = self.llm.with_structured_output(ExtractionOutput)
        # Allow exceptions to bubble up so caller (eval.py) can handle retries
        schema_out = await structured_llm.ainvoke(self.summary_prompt.format(dialogue=dialogue_str))
        
        # Convert to internal ReflectionSummary format
        summaries = []
        if schema_out and schema_out.extracted_memories:
            for em in schema_out.extracted_memories:
                # Provide default values for fields not in paper extraction but required by our schema
                # The paper extraction only gives summary and reference.
                # We can infer topic/entities/sentiment or leave generic.
                
                summaries.append(ReflectionSummary(
                    topic="General", # Paper prompt doesn't extract topic explicitly in this step
                    summary=em.summary,
                    entities=[],
                    sentiment="Neutral"
                ))
        return summaries

    async def consolidate(self, new_summaries: List[ReflectionSummary], user_id: Optional[str] = None, similarity_threshold: float = 0.85):
        """
        Check for existing memories that are similar to the new summaries.
        If similarity > threshold, merge/update. Else, add as new.
        """
        for summary in new_summaries:
            # Search for similar existing memories
            query = f"{summary.summary}"
            existing = self.memory_store.retrieve(query, k=1, user_id=user_id)

            if existing and existing[0].metadata.get("relevance_score", 0) > similarity_threshold:
                # MERGE LOGIC
                old_memory = existing[0]
                merged_content = f"{old_memory.content}\nUpdated: {summary.summary}"
                self.memory_store.update_memory(
                    memory_id=old_memory.id,
                    new_content=merged_content,
                    new_metadata={**old_memory.metadata, "updated": True, "user_id": user_id}
                )
            else:
                # ADD NEW
                entry = MemoryEntry(
                    id=summary.id,
                    content=f"{summary.summary}", # Just content for now as paper doesn't enforce specific format here 
                    metadata={
                        "topic": summary.topic,
                        "entities": summary.entities,
                        "sentiment": summary.sentiment,
                        "user_id": user_id
                    }
                )
                self.memory_store.add_memories([entry])
