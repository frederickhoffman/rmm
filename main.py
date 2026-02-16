import asyncio
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.memory_store import MemoryStore
from src.prospective import ProspectiveReflection
from src.retrospective import RetrospectiveReflection
from src.graph import RMMGraph

# Load environment variables
load_dotenv()

async def interactive_rmm():
    # Initialize components
    memory_store = MemoryStore()
    prospective = ProspectiveReflection(memory_store)
    retrospective = RetrospectiveReflection(memory_store)
    
    # Initialize Agent
    agent = RMMGraph(memory_store, prospective, retrospective)
    
    print("Welcome to the RMM Reflective Agent!")
    user_id = input("Enter User ID (default: 'admin'): ") or "admin"
    print(f"Logged in as: {user_id}")
    print("Type 'exit' to quit.")
    
    messages = []
    
    while True:
        user_input = input(f"\n[{user_id}] User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages.append(HumanMessage(content=user_input))
        
        # Invoke Agent
        inputs = {"messages": messages, "user_id": user_id}
        async for output in agent.graph.astream(inputs):
            # The agent outputs the full state after each node or as updates
            pass
        
        # Get the final response from the agent (last message in messages)
        # However, the state update via astream might need handling.
        # Simple invoke for CLI:
        final_state = await agent.graph.ainvoke(inputs)
        
        response = final_state["response"]
        citations = final_state["citations"]
        
        print(f"\nAssistant: {response}")
        if citations:
            print(f"(Citations used: {', '.join(citations)})")
        
        # update messages for context
        messages = final_state["messages"]

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file.")
    else:
        asyncio.run(interactive_rmm())
