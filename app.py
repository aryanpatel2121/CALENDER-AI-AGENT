import streamlit as st
import requests
from datetime import datetime
from typing import List, Dict
import pytz

# Configure the backend URL
BACKEND_URL = "https://google-calender-ai-agent.onrender.com"  # Update if your backend is hosted elsewhere

# Timezone setup
IST = pytz.timezone('Asia/Kolkata')  # Correct capitalization for timezone

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "tool_outputs" not in st.session_state:
    st.session_state.tool_outputs = []

# Custom CSS for better display
st.markdown("""
<style>
.event-card {
    border-left: 4px solid #4285F4;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f8f9fa;
    border-radius: 0.25rem;
}
.time-badge {
    background-color: #E8F0FE;
    color: #1967D2;
    padding: 0.25rem 0.5rem;
    border-radius: 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin-right: 0.5rem;
}
.assistant-message {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.title("📅 Google Calendar AI Assistant")
st.markdown("""
Welcome to your smart calendar assistant! I can:
- Show your upcoming events
- Find available time slots
- Schedule new meetings
- Answer questions about your schedule
""")

# Sidebar for conversation history
st.sidebar.title("Conversation History")
for msg in st.session_state.conversation_history:
    if msg["role"] == "user":
        st.sidebar.markdown(f"**You**: {msg['content']}")
    else:
        st.sidebar.markdown(f"**Assistant**: {msg['content']}")

# Function to parse and display events from response
def display_events(response_text: str):
    """Enhanced event display with custom formatting"""
    if "Here are your scheduled events" not in response_text:
        return st.markdown(response_text)
    
    # Split response into parts
    parts = response_text.split("\n\n")
    header = parts[0]
    events_text = "\n\n".join(parts[1:-1])  # Skip the last part which is usually a note
    
    # Display header
    st.markdown(f"### {header}")
    
    # Parse and display each event
    events = [e for e in events_text.split("\n") if e.strip()]
    for i in range(0, len(events), 2):
        if i+1 >= len(events):
            break
            
        time_line = events[i].strip()
        summary_line = events[i+1].strip()
        
        # Extract time and summary
        time_part = time_line.split("**")[1] if "**" in time_line else time_line
        summary = summary_line.split("**")[1] if "**" in summary_line else summary_line
        
        # Display as card
        st.markdown(f"""
        <div class="event-card">
            <div><span class="time-badge">{time_part}</span></div>
            <h4>{summary}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Display any notes
    if len(parts) > 2:
        st.info(parts[-1])

# Main chat interface
user_input = st.chat_input("Ask about your calendar (e.g. 'What's on my schedule tomorrow?')")

# Quick action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Today's Events"):
        user_input = "What events do I have today?"
with col2:
    if st.button("Tomorrow's Events"):
        user_input = "What events do I have tomorrow?"
with col3:
    if st.button("Find Available Slots"):
        user_input = "Show me available slots tomorrow between 9am and 5pm"

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.session_state.tool_outputs = []
    st.rerun()

# Instructions
with st.expander("💡 How to use this assistant"):
    st.markdown("""
    **Example commands:**
    - "What's on my calendar today?"
    - "Schedule a meeting with Alex tomorrow at 2pm for 1 hour"
    - "Find 30-minute slots available tomorrow"
    - "Do I have any conflicts on Friday?"
    
    **Features:**
    - Displays events in your local time (Asia/Kolkata)
    - Shows event durations clearly
    - Identifies scheduling conflicts
    - Provides direct calendar links for new events
    """)
    
if user_input:
    # Add user message to history
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Call the backend API
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={
                "message": user_input,
                "conversation_history": st.session_state.conversation_history[:-1],
                "debug": False
            },
            timeout=10  # Add timeout for API call
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Update session state
            st.session_state.conversation_history = data["conversation_history"]
            st.session_state.tool_outputs = data.get("tool_outputs", [])
            
            # Display assistant response
            with st.chat_message("assistant"):
                if "Here are your scheduled events" in data["response"]:
                    display_events(data["response"])
                else:
                    st.write(data["response"])
            
            # Show tool outputs if available
            if st.session_state.tool_outputs:
                with st.expander("🔧 Tool Execution Details"):
                    for tool in st.session_state.tool_outputs:
                        st.json(tool)
        else:
            error_msg = f"Backend error: {response.status_code} - {response.reason}"
            st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
    except requests.exceptions.RequestException as e:
        error_msg = f"Connection error: {str(e)}"
        st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.error(error_msg)

