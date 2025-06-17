import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any, List
import json

# Import your game API
from amdocs_game import GameAPI, GamePhase

# Page configuration
st.set_page_config(
    page_title="Corporate Conspiracy: The Amdocs Code",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling and UX with black, orange, white theme
st.markdown("""
<style>
    /* Global theme overrides */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: #1a1a1a;
    }
    
    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #ff6b35 100%);
        border: 2px solid #ff6b35;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: #ffffff;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: #ff6b35;
        font-size: 1.2rem;
        margin: 0;
        font-weight: 500;
    }
    
    /* Game container */
    .game-container {
        background-color: #000000;
        border: 2px solid #ff6b35;
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(255, 107, 53, 0.2);
    }
    
    /* Chat messages - HIGH CONTRAST */
    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 12px;
        border: 2px solid;
        font-size: 1.5rem;
        line-height: 1.6;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .user-message {
        background-color: #2a2a2a;
        border-color: #ff6b35;
        color: #ffffff;
        margin-left: 2rem;
    }
    
    .user-message strong {
        color: #ff6b35;
        font-weight: bold;
    }
    
    .system-message {
        background-color: #1a1a1a;
        border-color: #ffffff;
        color: #ffffff;
    }
    
    .victory-message {
        background-color: #0f2a0f;
        border-color: #4caf50;
        color: #ffffff;
        animation: victoryGlow 2s infinite alternate;
    }
    
    @keyframes victoryGlow {
        from { 
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4); 
            border-color: #4caf50;
        }
        to { 
            box-shadow: 0 8px 30px rgba(76, 175, 80, 0.8); 
            border-color: #66bb6a;
        }
    }
    
    /* Progress indicators */
    .progress-container {
        background-color: #2a2a2a;
        border: 2px solid #ff6b35;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    
    .progress-container h3 {
        color: #ff6b35;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1a1a1a;
        border: 2px solid #ff6b35;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffffff;
    }
    
    .metric-card h3 {
        color: #ff6b35;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
    }
    
    /* Phase indicators */
    .phase-indicator {
        padding: 1rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid;
        font-size: 1.1rem;
    }
    
    .phase-initial { 
        background-color: #1a1a1a; 
        color: #ff6b35; 
        border-color: #ff6b35;
    }
    .phase-investigating { 
        background-color: #2a2a2a; 
        color: #ffab00; 
        border-color: #ffab00;
    }
    .phase-ticker-discovered { 
        background-color: #1a2a1a; 
        color: #66bb6a; 
        border-color: #66bb6a;
    }
    .phase-decoding { 
        background-color: #1a1a2a; 
        color: #42a5f5; 
        border-color: #42a5f5;
    }
    .phase-near-solution { 
        background-color: #2a1a2a; 
        color: #ab47bc; 
        border-color: #ab47bc;
    }
    .phase-completed { 
        background-color: #0f2a0f; 
        color: #4caf50; 
        border-color: #4caf50;
        animation: phaseComplete 2s infinite alternate;
    }
    
    @keyframes phaseComplete {
        from { transform: scale(1); border-color: #4caf50; }
        to { transform: scale(1.05); border-color: #66bb6a; }
    }
    
    /* Input container */
    .input-container {
        position: sticky;
        bottom: 0;
        background-color: #000000;
        border: 2px solid #ff6b35;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 -4px 20px rgba(255, 107, 53, 0.3);
    }
    
    /* Discovery badges */
    .discovery-badge {
        display: inline-block;
        background-color: #ff6b35;
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: bold;
        margin: 0.3rem;
        border: 2px solid #ffffff;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Hint box */
    .hint-box {
        background-color: #2a2a1a;
        border: 3px dashed #ff6b35;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #ffffff;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Warning/Info messages */
    .warning-message {
        background-color: #2a1a1a;
        border-left: 6px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    .info-message {
        background-color: #1a1a2a;
        border-left: 6px solid #2196f3;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ffffff;
        font-size: 1.1rem;
    }
    
    /* Streamlit component overrides */
    .stTextInput > div > div > input {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 2px solid #ff6b35;
        border-radius: 8px;
        font-size: 1.1rem;
        padding: 0.75rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ffffff;
        box-shadow: 0 0 10px rgba(255, 107, 53, 0.5);
    }
    
    .stButton > button {
        background-color: #ff6b35;
        color: #000000;
        border: 2px solid #ffffff;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #ffffff;
        color: #ff6b35;
        border-color: #ff6b35;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.4);
    }
    
    .stProgress > div > div > div > div {
        background-color: #ff6b35;
    }
    
    .stProgress > div > div > div {
        background-color: #2a2a2a;
        border: 1px solid #ff6b35;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #2a2a2a;
        border: 2px solid #ff6b35;
        padding: 1rem;
        border-radius: 10px;
        color: #ffffff;
    }
    
    [data-testid="metric-container"] > div {
        color: #ffffff;
    }
    
    [data-testid="metric-container"] label {
        color: #ff6b35 !important;
        font-weight: bold;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2a2a2a;
        color: #ff6b35;
        border: 1px solid #ff6b35;
        border-radius: 8px;
        font-weight: bold;
    }
    
    .streamlit-expanderContent {
        background-color: #1a1a1a;
        border: 1px solid #ff6b35;
        border-top: none;
        color: #ffffff;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #2a2a2a;
        color: #ff6b35;
        border: 2px solid #ff6b35;
        font-weight: bold;
    }
    
    .stDownloadButton > button:hover {
        background-color: #ff6b35;
        color: #000000;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #0f2a0f;
        color: #4caf50;
        border: 2px solid #4caf50;
    }
    
    .stError {
        background-color: #2a0f0f;
        color: #f44336;
        border: 2px solid #f44336;
    }
    
    .stInfo {
        background-color: #0f0f2a;
        color: #2196f3;
        border: 2px solid #2196f3;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        border-top: 2px solid #ff6b35;
        margin-top: 3rem;
        color: #ff6b35;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize all session state variables"""
    if "game_api" not in st.session_state:
        st.session_state.game_api = GameAPI()
    
    if "game_session" not in st.session_state:
        st.session_state.game_session = st.session_state.game_api.start_new_game()
        # Initialize time from game session
        initial_time = st.session_state.game_session.get("time_remaining", 15)
        st.session_state.game_stats = {
            "score": 0,
            "attempts": 0,
            "phase": "initial",
            "discoveries": [],
            "time_remaining": initial_time,
            "game_duration": initial_time
        }
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        # Add initial game message to chat history
        st.session_state.chat_history.append({
            "type": "system",
            "message": st.session_state.game_session["message"],
            "timestamp": datetime.now().isoformat()
        })
    
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    
    if "input_history" not in st.session_state:
        st.session_state.input_history = []
    
    if "last_input_time" not in st.session_state:
        st.session_state.last_input_time = time.time()

def get_phase_display(phase: str) -> Dict[str, str]:
    """Get display information for game phases"""
    phase_info = {
        "initial": {"name": "🔍 Initial Investigation", "class": "phase-initial"},
        "investigating": {"name": "🕵️ Gathering Intel", "class": "phase-investigating"},
        "ticker_discovered": {"name": "📊 Market Analysis", "class": "phase-ticker-discovered"},
        "decoding": {"name": "🔢 Code Breaking", "class": "phase-decoding"},
        "near_solution": {"name": "🎯 Final Approach", "class": "phase-near-solution"},
        "completed": {"name": "✅ Mission Complete", "class": "phase-completed"}
    }
    return phase_info.get(phase, {"name": "🔍 Unknown Phase", "class": "phase-initial"})

def format_message(message: str, msg_type: str = "system") -> str:
    """Format messages with proper styling and high contrast"""
    css_class = f"{msg_type}-message chat-message"
    if "MISSION COMPLETE" in message or "ACCESS GRANTED" in message:
        css_class += " victory-message"
    
    # Ensure message content is properly formatted for readability
    formatted_message = message.replace('\n', '<br>')
    
    return f'<div class="{css_class}">{formatted_message}</div>'

def display_game_header():
    """Display the main game header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔐 CORPORATE CONSPIRACY: THE AMDOCS CODE</h1>
        <p>Uncover the truth. Break the code. Save the evidence.</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display the game sidebar with stats and controls"""
    with st.sidebar:
        st.markdown("### 📊 Mission Status")
        
        # Game metrics with better styling
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="🎯 Score", 
                value=f"{st.session_state.game_stats['score']}/100",
                help="Your investigation progress"
            )
        with col2:
            st.metric(
                label="🔄 Attempts", 
                value=st.session_state.game_stats["attempts"],
                help="Number of inputs made"
            )
        
        # Phase metric
        st.metric(
            label="🎯 Phase", 
            value=st.session_state.game_stats["phase"].title(),
            help="Current investigation phase"
        )
        
        # Phase indicator
        phase_info = get_phase_display(st.session_state.game_stats["phase"])
        st.markdown(f"""
        <div class="phase-indicator {phase_info['class']}">
            {phase_info['name']}
        </div>
        """, unsafe_allow_html=True)
        
        # Discoveries
        if st.session_state.game_stats["discoveries"]:
            st.markdown("### 🎯 Key Discoveries")
            for discovery in st.session_state.game_stats["discoveries"]:
                st.markdown(f'<span class="discovery-badge">{discovery}</span>', 
                          unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Game controls
        st.markdown("### 🎮 Game Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Game", help="Start fresh investigation", use_container_width=True):
                restart_game()
        
        with col2:
            if st.button("💡 Hint", help="Get help if stuck", use_container_width=True):
                request_hint()
        
        if st.button("📋 Export Log", help="Download investigation history", use_container_width=True):
            export_chat_history()
        
        # Quick help
        with st.expander("❓ Quick Help", expanded=False):
            st.markdown("""
            **🎯 How to Play:**
            - Ask questions about Amdocs
            - Look for clues in responses  
            - Try different approaches
            - The code is hidden in plain sight
            
            **💡 Tips:**
            - Think about financial markets
            - Consider company identifiers
            - Numbers can be converted
            - Read instructions carefully
            
            **🔍 Commands:**
            - Type questions naturally
            - Ask for hints when stuck
            - Try company/stock queries
            """)
        
        # Session info
        with st.expander("ℹ️ Session Details"):
            session_info = st.session_state.game_api.get_session_info(
                st.session_state.game_session["session_id"]
            )
            if "error" not in session_info:
                st.json(session_info)

def display_chat_interface():
    """Display the main chat interface with high contrast"""
    st.markdown('<div class="game-container">', unsafe_allow_html=True)
    
    # Chat history container with scrolling
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="info-message">
                <strong>🚀 Welcome, Agent!</strong><br>
                Start your investigation by asking questions about Amdocs or making your first guess.
            </div>
            """, unsafe_allow_html=True)
        
        # Display all messages with high contrast
        for i, chat_msg in enumerate(st.session_state.chat_history):
            if chat_msg["type"] == "user":
                st.markdown(
                    format_message(f"{chat_msg['message']}", "user"), 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    format_message(f"{chat_msg['message']}", "system"), 
                    unsafe_allow_html=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_input_interface():
    """Display the input interface with enhanced UX"""
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Main input form
    with st.form("user_input_form", clear_on_submit=True):
        st.markdown("### 💬 Your Investigation")
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                label="Type your question or guess:",
                placeholder="Ask about Amdocs, make guesses, or request hints...",
                help="Enter your questions or code guesses here",
                key="current_input",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button(
                "🔍 Send", 
                use_container_width=True,
                help="Send your message"
            )
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            hint_pressed = st.form_submit_button("💡 Need Hint", help="Request a hint")
        with col2:
            help_pressed = st.form_submit_button("❓ General Help", help="Get general guidance")
        with col3:
            company_pressed = st.form_submit_button("🏢 About Company", help="Learn about Amdocs")
        with col4:
            stock_pressed = st.form_submit_button("📈 Stock Info", help="Get market data")
        
        # Handle quick actions
        if hint_pressed:
            user_input = "I need a hint, please help me"
            submit_button = True
        elif help_pressed:
            user_input = "How do I solve this puzzle?"
            submit_button = True
        elif company_pressed:
            user_input = "Tell me about Amdocs company"
            submit_button = True
        elif stock_pressed:
            user_input = "What's Amdocs stock symbol and market information?"
            submit_button = True
    
    # Process input
    if submit_button and user_input and user_input.strip():
        process_user_input(user_input.strip())
    elif submit_button and not user_input.strip():
        st.warning("⚠️ Please enter a question or guess before sending.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def process_user_input(user_input: str):
    """Process user input and update game state"""
    try:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "message": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to input history for quick access
        if user_input not in st.session_state.input_history:
            st.session_state.input_history.insert(0, user_input)
            st.session_state.input_history = st.session_state.input_history[:10]  # Keep last 10
        
        # Process with game API
        with st.spinner("🔍 Processing your investigation..."):
            response = st.session_state.game_api.process_message(
                st.session_state.game_session["session_id"],
                user_input
            )
        
        # Handle errors
        if "error" in response:
            st.error(f"❌ Error: {response['error']}")
            return
        
        # Add response to chat history
        st.session_state.chat_history.append({
            "type": "system",
            "message": response["message"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Update game stats
        if "metadata" in response:
            st.session_state.game_stats.update({
                "score": response["metadata"]["score"],
                "attempts": response["metadata"]["attempts"],
                "phase": response["metadata"]["phase"],
                "completed": response["metadata"]["completed"]
            })
            
            # Get latest session info for discoveries and time
            session_info = st.session_state.game_api.get_session_info(
                st.session_state.game_session["session_id"]
            )
            if "error" not in session_info:
                st.session_state.game_stats["discoveries"] = session_info.get("discoveries", [])
                # Only update time if it's not None and greater than 0
                new_time = session_info.get("time_remaining")
                if new_time is not None and new_time >= 0:
                    st.session_state.game_stats["time_remaining"] = new_time
        
        # Check for completion or time up
        if response.get("state") == "completed":
            if st.session_state.game_stats["time_remaining"] <= 0:
                st.error("⏰ **TIME'S UP!** The server remains locked. Better luck next time!")
            else:
                st.balloons()
                st.success("🎉 **MISSION ACCOMPLISHED!** You've successfully cracked the code!")
        
        # Auto-scroll and refresh
        st.session_state.last_input_time = time.time()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Try rephrasing your question or starting a new game.")

def restart_game():
    """Restart the game with fresh state"""
    # Clear relevant session state
    keys_to_clear = [k for k in st.session_state.keys() 
                     if k.startswith(('game_', 'chat_', 'user_', 'input_', 'last_'))]
    
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()
    st.success("🔄 **New investigation started!** Good luck, Agent.")
    st.rerun()

def request_hint():
    """Request a hint from the game"""
    process_user_input("I need a hint, please help me")

def export_chat_history():
    """Export chat history as downloadable file"""
    try:
        chat_data = {
            "session_id": st.session_state.game_session["session_id"],
            "export_time": datetime.now().isoformat(),
            "game_stats": st.session_state.game_stats,
            "chat_history": st.session_state.chat_history
        }
        
        json_str = json.dumps(chat_data, indent=2)
        
        st.download_button(
            label="📥 Download Investigation Log",
            data=json_str,
            file_name=f"amdocs_investigation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download your complete investigation history"
        )
        
    except Exception as e:
        st.error(f"❌ Failed to export chat history: {str(e)}")

def display_progress_indicators():
    """Display visual progress indicators with high contrast"""
    if st.session_state.game_stats["score"] > 0 or st.session_state.game_stats["attempts"] > 0:
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Investigation Progress")
            progress_value = min(st.session_state.game_stats["score"] / 100, 1.0)
            st.progress(progress_value)
            st.markdown(f"**{st.session_state.game_stats['score']}/100 points**")
        
        with col2:
            st.markdown("### ⏱️ Time Remaining")
            time_remaining = st.session_state.game_stats["time_remaining"]
            time_progress = time_remaining / st.session_state.game_stats["game_duration"]
            st.progress(time_progress)
            st.markdown(f"**{time_remaining} minutes left**")
        
        with col3:
            st.markdown("### 🔍 Discoveries Made")
            discovery_count = len(st.session_state.game_stats["discoveries"])
            st.markdown(f"**{discovery_count} key insights found**")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_game_header()
    
    # Display progress indicators
    display_progress_indicators()
    
    # Main layout
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Chat interface
        display_chat_interface()
        
        # Input interface
        display_input_interface()
        
        # Show recent inputs for quick access
        if st.session_state.input_history:
            with st.expander("🔄 Recent Questions", expanded=False):
                st.markdown("**Click to reuse a previous question:**")
                for i, recent_input in enumerate(st.session_state.input_history[:5]):
                    if st.button(
                        f"↻ {recent_input[:60]}{'...' if len(recent_input) > 60 else ''}", 
                        key=f"recent_{i}",
                        help=f"Reuse: {recent_input}"
                    ):
                        process_user_input(recent_input)
    
    with col2:
        # Sidebar content
        display_sidebar()
    
    # # Footer
    # st.markdown("""
    # <div class="footer">
    #     🔐 Corporate Conspiracy Game | Enterprise-Grade Mystery Experience
    # </div>
    # """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()