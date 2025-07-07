from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import parse_qs
from pytz import UTC
import pytz
from typing_extensions import TypedDict, Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langgraph.graph import END, START
import json

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    return {"message": "Calendar AI Agent is live. Use /chat to interact."}

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CALENDAR_ID = os.getenv("CALENDAR_ID")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "google_calender")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
# Google Calendar setup
SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    if not GOOGLE_SERVICE_ACCOUNT_JSON:
        raise Exception("Missing GOOGLE_SERVICE_ACCOUNT_JSON env variable")
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(info, SCOPES)
    return build('calendar', 'v3', credentials=credentials)

calendar_service = get_calendar_service()

# Define tools (same as in your original script)
@tool()
def today(tool_input: dict = None) -> dict:
    """Returns today's date and day in UTC."""
    now = datetime.now(UTC)
    return {
        "role": "system",
        "content": f"Today's date is {now.strftime('%Y-%m-%d')}, and it is a {now.strftime('%A')} (UTC)."
    }

@tool()
def list_events_tool(query: str = None) -> list:
    """
    Fetches events based on time range parameters.
    Format: "start_time=ISO_UTC_TIME&end_time=ISO_UTC_TIME"
    Returns list of events or message if none found.
    """
    # Default time range (now to 7 days later)
    now = datetime.now(UTC)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=7)).isoformat()
    
    # Parse query parameters if provided
    if query and '=' in query:
        try:
            params = {k: v[0] for k, v in parse_qs(query).items()}
            time_min = params.get('start_time', time_min)
            time_max = params.get('end_time', time_max)
        except:
            return "Invalid query format. Use: start_time=ISO_TIME&end_time=ISO_TIME"
    
    # Fetch events
    try:
        events_result = calendar_service.events().list(
            calendarId='patelaryan2006k@gmail.com',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=20,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        # Format and deduplicate events
        unique_events = []
        seen = set()
        for event in events:
            event_key = (
                event['start'].get('dateTime', event['start'].get('date')),
                event['summary']
            )
            if event_key not in seen:
                seen.add(event_key)
                unique_events.append({
                    'start': event['start'].get('dateTime', event['start'].get('date')),
                    'summary': event['summary'],
                    'end': event['end'].get('dateTime', event['end'].get('date'))
                })
        
        if not unique_events:
            return f"No events found between {time_min} and {time_max}"
        
        return unique_events
    
    except Exception as e:
        return {"error": f"Error fetching events: {str(e)}"}  # Return error as a dictionary

@tool()
def find_available_slots_tool(min_duration: int = 30) -> list:
    """
    Finds available time slots of at least min_duration minutes.
    Returns list of slots in UTC.
    """
    now = datetime.now(UTC)
    events = calendar_service.events().list(
        calendarId='patelaryan2006k@gmail.com',
        timeMin=now.isoformat(),
        maxResults=20,
        singleEvents=True,
        orderBy='startTime'
    ).execute().get('items', [])
    
    available_slots = []
    last_end = now
    
    for event in events:
        start = datetime.fromisoformat(event['start']['dateTime']).astimezone(UTC)
        end = datetime.fromisoformat(event['end']['dateTime']).astimezone(UTC)
        
        if start > last_end + timedelta(minutes=min_duration):
            available_slots.append({
                'start': last_end.isoformat(),
                'end': start.isoformat(),
                'duration_minutes': int((start - last_end).total_seconds() / 60)
            })
        last_end = max(last_end, end)
    
    # Add remaining time until end of day
    end_of_day = now.replace(hour=23, minute=59, second=59)
    if end_of_day > last_end + timedelta(minutes=min_duration):
        available_slots.append({
            'start': last_end.isoformat(),
            'end': end_of_day.isoformat(),
            'duration_minutes': int((end_of_day - last_end).total_seconds() / 60)
        })
    
    return available_slots

@tool()
def create_event_tool(summary: str, start_time: str, end_time: str) -> dict:
    """
    Creates a calendar event in UTC.
    Args:
        summary: Event title
        start_time: ISO format in UTC
        end_time: ISO format in UTC
    """
    try:
        # Validate times
        start = datetime.fromisoformat(start_time).astimezone(UTC)
        end = datetime.fromisoformat(end_time).astimezone(UTC)
        
        if start >= end:
            return {"error": "End time must be after start time"}
        if start < datetime.now(UTC):
            return {"error": "Cannot create events in the past"}
        
        event = {
            'summary': summary,
            'start': {'dateTime': start.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end.isoformat(), 'timeZone': 'UTC'}
        }
        
        created_event = calendar_service.events().insert(
            calendarId='patelaryan2006k@gmail.com',
            body=event
        ).execute()
        
        return {
            "status": "success",
            "link": created_event.get("htmlLink"),
            "event_id": created_event.get("id")
        }
    
    except Exception as e:
        return {"error": f"Error creating event: {str(e)}"}  # Return error as a dictionary

# LangGraph setup
class State(TypedDict):
    messages: Annotated[List[SystemMessage | HumanMessage], add_messages]
    today_info: dict  # Injected by the `today` tool

# Initialize LLM and tools
llm = init_chat_model("groq:qwen-qwq-32b")
tools = [today, list_events_tool, find_available_slots_tool, create_event_tool]
llm_with_tools = llm.bind_tools(tools)

# Helper functions
def provide_today_info(state: State):
    """Inject today's date (and weekday) into the state."""
    statestate["today_info"] = today.invoke({})
    return state

def inject_additional_system_message(state: State):
    """Add the scheduling‐role system message into the conversation."""
    sys_msg = SystemMessage(
        content="""Your role is to schedule events based on user instructions. Follow these rules:
        1. Firstly check the todays date by using the above node called provide_today_info.
        2. Parse user instructions to extract the date, time range, and description.
        3. Assume the user's specified time is in their local timezone. For this workflow, use 'Asia/Kolkata' (UTC+5:30) as the default timezone.
        4. Convert the extracted date and time into ISO 8601 UTC format.
        5. ALWAYS CALL THE TOOLS IN ISO 8601 UTC FORMAT ONLY.Now to call tools in the following order:
        - `list_events_tool`: To check for existing events on the specified date.
        - `find_available_slots_tool`: To find gaps between events (optional if the user provides a specific time).
        - `create_event_tool`: To schedule the meeting, using the validated date and time.
        6. Use the current date provided by the system to calculate dates like tomorrow or next week etc.
        7. Provide the final output in precise and correct manner."""
    )
    state["messages"].append(sys_msg)
    return state

def call_llm_model(state: State):
    """Invoke the model with the accumulated messages (including tools)."""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the LangGraph
graph_builder = StateGraph(State)
graph_builder.add_node("inject_today_info", provide_today_info)
graph_builder.add_node("inject_additional_message", inject_additional_system_message)
graph_builder.add_node("process_input", call_llm_model)

# Tool‐execution node
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Edges
graph_builder.add_edge(START, "inject_today_info")
graph_builder.add_edge("inject_today_info", "inject_additional_message")
graph_builder.add_edge("inject_additional_message", "process_input")

graph_builder.add_conditional_edges(
    "process_input",
    tools_condition  # routes to tools node when needed, else to END
)
graph_builder.add_edge("tools", "process_input")

graph = graph_builder.compile()

# API Models
class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Dict[str, str]] = []

class ChatResponse(BaseModel):
    response: str
    conversation_history: List[Dict[str, str]]
    tool_outputs: List[Dict[str, Any]] = []

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_calendar(request: ChatRequest):
    try:
        # Convert conversation history to LangChain messages
        messages = []
        for msg in request.conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(SystemMessage(content=msg["content"]))
        
        # Add the new user message
        messages.append(HumanMessage(content=request.message))
        
        # Run the graph
        state = {"messages": messages}
        result = graph.invoke(state)
        
        # Extract the response
        last_message = result["messages"][-1].content
        tool_outputs = []
        
        # Check for tool outputs in the result
        if hasattr(result["messages"][-1], 'tool_calls'):
            for tool_call in result["messages"][-1].tool_calls:
                tool_outputs.append({
                    "tool_name": tool_call['name'],
                    "input": tool_call['args'],
                    "output": None  # You might want to capture actual tool outputs
                })
        
        # Update conversation history
        updated_history = request.conversation_history.copy()
        updated_history.append({"role": "user", "content": request.message})
        updated_history.append({"role": "assistant", "content": last_message})
        
        return ChatResponse(
            response=last_message,
            conversation_history=updated_history,
            tool_outputs=tool_outputs
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
