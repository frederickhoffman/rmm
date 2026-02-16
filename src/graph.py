from typing import Annotated, List, TypedDict, Dict, Any, Union
import re
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.memory_store import MemoryStore, MemoryEntry
from src.prospective import ProspectiveReflection
from src.retrospective import RetrospectiveReflection

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_id: str
    memories: List[MemoryEntry]
    scores: List[float]
    response: str
    citations: List[str]
    loss: float

class RMMGraph:
    def __init__(self, memory_store: MemoryStore, prospective: ProspectiveReflection, retrospective: RetrospectiveReflection):
        self.memory_store = memory_store
        self.prospective = prospective
        self.retrospective = retrospective
        self.llm = ChatOpenAI(model="gpt-4o")
        
        # Build the graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("respond", self.respond_node)
        workflow.add_node("update_weights", self.update_weights_node)
        workflow.add_node("reflect", self.reflect_node)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", "update_weights")
        workflow.add_edge("update_weights", "reflect")
        workflow.add_edge("reflect", END)
        
        self.graph = workflow.compile()

    async def retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1].content
        user_id = state.get("user_id", "default_user")
        # Initial semantic search
        memories = self.memory_store.retrieve(last_message, k=5, user_id=user_id)
        # Neuronal re-ranking
        sorted_memories, scores = self.retrospective.rerank(last_message, memories)
        
        return {
            "memories": sorted_memories,
            "scores": scores.tolist() if hasattr(scores, "tolist") else []
        }

    async def respond_node(self, state: AgentState) -> Dict[str, Any]:
        last_message = state["messages"][-1].content
        memories = state["memories"]
        
        # Format memories for the prompt
        # Paper format:
        # – Memory [0]: <summary>
        #     * <turn> (if available, detailed content)
        formatted_memories = ""
        for i, m in enumerate(memories):
            formatted_memories += f"        – Memory [{i}]: {m.content}\n"
        
        prompt = f"""Task Description: Given a user query and a list of memories consisting of personal summaries with their corresponding original turns, generate a natural and fluent response while adhering to the following guidelines:
   • Cite useful memories using [ i], where i corresponds to the index of the cited memory.
   • Do not cite memories that are not useful. If no useful memory exist, output [NO_CITE].
   • Each memory is independent and may repeat or contradict others. The response must be directly supported by cited memories.
   • If the response relies on multiple memories, list all corresponding indices, e.g., [ i, j, k].
   • The citation is evaluated based on whether the response references the original turns, not the summaries.

Examples:
Case 1: Useful Memories Found
INPUT:
   • User Query: SPEAKER_1: What hobbies do I enjoy?
   • Memories:
        – Memory [0]: SPEAKER_1 enjoys hiking and often goes on weekend trips.
            * Speaker 1: I love spending my weekends hiking in the mountains.
              Speaker 2: That sounds amazing! Do you go alone or with friends?
            * Speaker 1: Last month, I hiked a new trail and it was amazing.
              Speaker 2: Nice! Which trail was it?
        – Memory [1]: SPEAKER_1 plays the guitar and occasionally performs at open mics.
            * Speaker 1: I’ve been practicing guitar for years and love playing at open mics.
              Speaker 2: That’s awesome! What songs do you usually play?
            * Speaker 1: I performed at a local cafe last week and had a great time.
              Speaker 2: That must have been fun! Were there a lot of people?
        – Memory [2]: SPEAKER_1 is interested in astronomy and enjoys stargazing.
            * Speaker 1: I recently bought a telescope to get a closer look at planets.
              Speaker 2: That’s so cool! What have you seen so far?
            * Speaker 1: I love stargazing, especially when there’s a meteor shower.
              Speaker 2: I’d love to do that sometime. When’s the next one?
Output: You enjoy hiking, playing the guitar, and stargazing. [0, 1, 2]

Case 2: No Useful Memories
INPUT:
   • User Query: SPEAKER_1: What countries did I go to last summer?
   • Memories:
        – Memory [0]: SPEAKER_1 enjoys hiking and often goes on weekend trips.
            * Speaker 1: I love spending my weekends hiking in the mountains.
              Speaker 2: That sounds amazing! Do you go alone or with friends?
            * Speaker 1: Last month, I hiked a new trail and it was amazing.
              Speaker 2: Nice! Which trail was it?
Output: I don't have enough information to answer that. [NO_CITE]

Additional Instructions:
   • Ensure the response is fluent and directly answers the user's query.
   • Always cite the useful memory indices explicitly.
   • The citation is evaluated based on whether the response references the original turns, not the summaries.
   • Format citations as [0], [1, 2] etc.

Input:
   • User Query: {last_message}
   • Memories:
{formatted_memories}
Output:"""
        
        response = await self.llm.ainvoke(prompt)
        
        # Extract citations
        # Paper format: [0, 1, 2] or [0]
        # We need to parse this somewhat loosely to be robust
        citations = re.findall(r"\[([\d,\s]+)\]", response.content)
        parsed_citations = []
        for c in citations:
            if "NO_CITE" in c:
                continue
            # Split by comma
            nums = c.split(",")
            for n in nums:
                n = n.strip()
                if n.isdigit():
                    # Reconstruct as [Memory X] for consistency with internal logic if needed, 
                    # but current logic expects list of strings? 
                    # update_weights_node expects state["citations"] passed to retrospective.update_weights
                    # retrospective.py likely iterates them.
                    # Let's see what retrospective expects.
                    # Assuming it handles indices. The previous regex was re.findall(r"\[Memory \d+\]", ...)
                    # So it expected "[Memory 1]".
                    # Now we get "[0]".
                    # We should normalize to "[Memory X]" or update retrospective.
                    # For minimal friction, I will normalize here to "[Memory X]" where X is the index.
                    parsed_citations.append(f"[Memory {n}]")
        
        return {
            "response": response.content,
            "citations": parsed_citations,
            "messages": [AIMessage(content=response.content)]
        }

    async def update_weights_node(self, state: AgentState) -> Dict[str, Any]:
        last_human_message = ""
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                last_human_message = m.content
                break
        
        loss = self.retrospective.update_weights(
            query=last_human_message,
            memories=state["memories"],
            citations=state["citations"]
        )
        return {"loss": loss}

    async def reflect_node(self, state: AgentState) -> Dict[str, Any]:
        # Prospective Reflection: Summarize the recent interaction
        user_id = state.get("user_id", "default_user")
        recent_messages = state["messages"][-2:] # Last pair
        summaries = await self.prospective.summarize_dialogue(recent_messages)
        await self.prospective.consolidate(summaries, user_id=user_id)
        return {}

# Default instances for LangGraph Server / Studio
_memory_store = MemoryStore()
_prospective = ProspectiveReflection(_memory_store, model_name="gpt-4o-mini")
_retrospective = RetrospectiveReflection(_memory_store)
_rmm_graph = RMMGraph(_memory_store, _prospective, _retrospective)
compiled_graph = _rmm_graph.graph
