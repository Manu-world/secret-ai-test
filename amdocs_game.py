import os
import json
import logging
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Annotated, Any, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from operator import add
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============= Configuration Management =============
@dataclass
class GameConfig:
    """Centralized game configuration"""
    secret_code: str = "41524"
    company_name: str = "Amdocs"
    ticker_symbol: str = "DOX"
    max_attempts: int = 50
    hint_threshold_scores: Dict[str, int] = field(default_factory=lambda: {
        "hint_1": 30,
        "hint_2": 50,
        "hint_3": 70
    })
    
    # LLM configurations
    primary_model: str = "gpt-4o-mini"
    scoring_model: str = "gpt-4o-mini"
    primary_temperature: float = 0.8
    scoring_temperature: float = 0.1
    
    # Integration settings
    enable_analytics: bool = True
    enable_persistence: bool = True
    api_timeout: int = 30

# ============= Game State Management =============
class GamePhase(Enum):
    """Game progression phases"""
    INITIAL = "initial"
    INVESTIGATING = "investigating"
    TICKER_DISCOVERED = "ticker_discovered"
    DECODING = "decoding"
    NEAR_SOLUTION = "near_solution"
    COMPLETED = "completed"

class GameState(TypedDict):
    """Complete game state definition"""
    # Core state
    messages: Annotated[List[BaseMessage], add]
    phase: str
    score: int
    attempts: int
    
    # User interaction
    current_input: str
    feedback: str
    
    # Tracking
    discoveries: List[str]
    hints_revealed: List[str]
    wrong_attempts: List[Dict[str, Any]]
    previous_guesses: List[str]
    
    # Analytics
    session_id: str
    start_time: str
    end_time: Optional[str]
    
    # Internal processing
    similarity_score: int
    is_hint_request: bool
    needs_hint: bool
    is_correct: bool

# ============= Prompt Templates =============
class PromptTemplates:
    """Centralized prompt management for consistency"""
    
    MAIN_SYSTEM = """You are an anonymous contact helping a hacker uncover a corporate conspiracy at {company_name}. 
    The whistleblower's message was: "Convert the symbol to numbers and read them in order."

    Current Phase: {phase}
    Score: {score}
    Discoveries Made: {discoveries}

    Guidelines:
    1. Keep responses brief and direct
    2. Only provide guidance when explicitly asked
    3. For guesses, simply indicate if they're correct or not
    4. For wrong guesses of code or number, you can ask how they arrived at that guess
    5. For red herrings (addition/dates), correct the misunderstanding
    6. Never reveal the answer ({secret}) directly
    7. Never provide unsolicited hints

    Example responses(not limited to these):
    - For questions: "Amdocs trades on NASDAQ with the ticker DOX."
    - For wrong guesses: "That's not it. But how did you arrive at that?"
    - For addition attempts: "It's a sequence, not a sum."
    - For date attempts: "It's not a date, it's a sequence of numbers."

    Respond as a mysterious guide would - direct and to the point."""
    
    SCORING_SYSTEM = """Analyze how close this guess/question is to solving the Amdocs conspiracy puzzle.
    
    Solution: The code is {secret}, derived from ticker "DOX" (D=4, O=15, X=24, read as 41524)
    Current Total Score: {current_score}
    Current Phase: {phase}
    
    User Input: {input}
    Previous Guesses: {previous_guesses}
    
    CRITICAL RULES:
    1. If the user input contains the EXACT secret code {secret}, return score_delta to make total score 100
    2. If this exact input was guessed before, return score_delta: 0 (no points for repeating)
    3. Maximum score without finding the secret code is 95
    4. Score based on progress toward solution:
       - Asking about {company_name}/NASDAQ/ticker: +5-10
       - Mentioning DOX: +15-20
       - Attempting number conversion: +10-15
       - Getting partial code right: +20-30
       - Wrong direction (addition/dates): -5 to -10
       - Unrelated queries: -2
    
    Calculate score_delta to add to current score of {current_score}.
    
    Return ONLY valid JSON:
    {{"score_delta": X, "reasoning": "brief explanation", "phase_change": "new_phase_or_null", "is_correct_answer": false}}
    
    If input contains exact code {secret}, set is_correct_answer to true and score_delta to make total 100."""
    
    HINT_TEMPLATES = {
        "hint_1": "🔍 'Financial markets hold many secrets. Companies trade under symbols...'",
        "hint_2": "📊 'NASDAQ knows Amdocs by three letters. Each letter has a position...'",
        "hint_3": "🔤 'D-O-X. Fourth, fifteenth, twenty-fourth. Read them as instructed.'"
    }

# ============= Service Layer =============
class LLMService:
    """Manages all LLM interactions"""
    
    def __init__(self, config: GameConfig):
        self.config = config
        self.primary_llm = ChatOpenAI(
            model=config.primary_model,
            temperature=config.primary_temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=config.api_timeout
        )
        self.scoring_llm = ChatOpenAI(
            model=config.scoring_model,
            temperature=config.scoring_temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=config.api_timeout
        )
    
    def get_response(self, messages: List[BaseMessage]) -> str:
        """Get LLM response with error handling"""
        try:
            response = self.primary_llm.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"LLM response error: {e}")
            return "Connection unstable... Try again."
    
    def get_score(self, prompt: str) -> Dict[str, Any]:
        """Get scoring analysis with validation"""
        try:
            response = self.scoring_llm.invoke([SystemMessage(content=prompt)])
            result = json.loads(response.content)
            
            # Validate response structure
            assert "score_delta" in result
            assert isinstance(result["score_delta"], (int, float))
            
            return result
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return {
                "score_delta": 0, 
                "reasoning": "Error in analysis", 
                "phase_change": None,
                "is_correct_answer": False
            }

# ============= Game Logic Components =============
class GameAnalyzer:
    """Analyzes game state and player progress"""
    
    @staticmethod
    def check_win_condition(user_input: str, config: GameConfig) -> bool:
        """Check if player has found the solution"""
        # Clean and normalize input
        cleaned_input = user_input.strip().lower()
        
        # Check for direct matches
        if config.secret_code in cleaned_input:
            return True
            
        # Check for variations like "is it 41524?", "the code 41524", etc.
        if any(pattern in cleaned_input for pattern in [
            f"is it {config.secret_code}",
            f"the code {config.secret_code}",
            f"code {config.secret_code}",
            f"number {config.secret_code}",
            f"answer {config.secret_code}",
            f"solution {config.secret_code}"
        ]):
            return True
            
        # Extract numbers from the input
        import re
        numbers = re.findall(r'\d+', cleaned_input)
        
        # Check if any extracted number matches the secret code
        return any(num == config.secret_code for num in numbers)
    
    @staticmethod
    def is_repeat_guess(user_input: str, previous_guesses: List[str]) -> bool:
        """Check if this is a repeated guess"""
        cleaned_input = user_input.strip().lower()
        
        for prev_guess in previous_guesses:
            if cleaned_input == prev_guess.strip().lower():
                return True
        
        return False
    
    @staticmethod
    def detect_hint_request(user_input: str) -> bool:
        """Detect if user is asking for help"""
        hint_indicators = ["hint", "help", "stuck", "clue", "guide", "lost", "confused"]
        return any(indicator in user_input.lower() for indicator in hint_indicators)
    
    @staticmethod
    def detect_key_discoveries(user_input: str, state: GameState, config: GameConfig) -> List[str]:
        """Track important discoveries made by the player"""
        discoveries = []
        input_lower = user_input.lower()
        
        if config.company_name.lower() in input_lower and "NASDAQ" not in state["discoveries"]:
            if "nasdaq" in input_lower or "stock" in input_lower or "market" in input_lower:
                discoveries.append("NASDAQ")
        
        if config.ticker_symbol.lower() in input_lower and "TICKER" not in state["discoveries"]:
            discoveries.append("TICKER")
        
        if any(phrase in input_lower for phrase in ["alphabet", "position", "letter to number", "a=1"]):
            if "CONVERSION" not in state["discoveries"]:
                discoveries.append("CONVERSION")
        
        return discoveries
    
    @staticmethod
    def determine_phase(discoveries: List[str], score: int) -> str:
        """Determine current game phase based on progress"""
        if "TICKER" in discoveries and "CONVERSION" in discoveries:
            return GamePhase.NEAR_SOLUTION.value
        elif "TICKER" in discoveries:
            return GamePhase.DECODING.value
        elif "NASDAQ" in discoveries:
            return GamePhase.TICKER_DISCOVERED.value
        elif score > 20:
            return GamePhase.INVESTIGATING.value
        else:
            return GamePhase.INITIAL.value

# ============= LangGraph Node Implementations =============
class AmdocsConspiracyGame:
    """Main game engine using LangGraph"""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self.llm_service = LLMService(self.config)
        self.analyzer = GameAnalyzer()
        self.templates = PromptTemplates()
        self.graph = self._build_graph()
        
        logger.info("Amdocs Conspiracy Game initialized")
    
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow"""
        workflow = StateGraph(GameState)
        
        # Add all nodes with descriptive names
        workflow.add_node("validate_and_prepare_input", self.preprocess_input)
        workflow.add_node("check_solution_correctness", self.check_win_condition)
        workflow.add_node("evaluate_player_progress", self.analyze_input)
        workflow.add_node("update_game_progress", self.update_game_state)
        workflow.add_node("generate_game_response", self.generate_response)
        workflow.add_node("handle_successful_completion", self.handle_victory)
        workflow.add_node("provide_guided_hint", self.provide_hint)
        
        # Define the flow
        workflow.set_entry_point("validate_and_prepare_input")
        workflow.add_edge("validate_and_prepare_input", "check_solution_correctness")
        
        # Conditional routing based on win condition
        workflow.add_conditional_edges(
            "check_solution_correctness",
            lambda x: "handle_successful_completion" if x["is_correct"] else "evaluate_player_progress",
            {
                "handle_successful_completion": "handle_successful_completion",
                "evaluate_player_progress": "evaluate_player_progress"
            }
        )
        
        workflow.add_edge("evaluate_player_progress", "update_game_progress")
        
        # Conditional routing for hints
        workflow.add_conditional_edges(
            "update_game_progress",
            lambda x: "provide_guided_hint" if x.get("needs_hint", False) else "generate_game_response",
            {
                "provide_guided_hint": "provide_guided_hint",
                "generate_game_response": "generate_game_response"
            }
        )
        
        workflow.add_edge("provide_guided_hint", END)
        workflow.add_edge("generate_game_response", END)
        workflow.add_edge("handle_successful_completion", END)
        
        # graph=workflow.compile()
        # mermaid_png=graph.get_graph().draw_mermaid_png()
        # with open("workflow.png", "wb") as f:
        #     f.write(mermaid_png)
        
        return workflow.compile()
    
    def preprocess_input(self, state: GameState) -> GameState:
        """Initial input processing and validation"""
        state["current_input"] = state["current_input"].strip()
        state["attempts"] = state.get("attempts", 0) + 1
        
        # Detect hint requests
        state["is_hint_request"] = self.analyzer.detect_hint_request(state["current_input"])
        
        logger.info(f"Processing input: {state['current_input'][:50]}...")
        return state
    
    def check_win_condition(self, state: GameState) -> GameState:
        """Check if the player has found the solution"""
        # First check if it's a win condition
        state["is_correct"] = self.analyzer.check_win_condition(
            state["current_input"], 
            self.config
        )
        
        # If correct, immediately set score to 100
        if state["is_correct"]:
            state["score"] = 100
            logger.info(f"WIN CONDITION MET! Score set to 100")
        
        return state
    
    def analyze_input(self, state: GameState) -> GameState:
        """Deep analysis of user input for scoring and progress tracking"""
        # Check if this is a repeated guess
        previous_guesses = state.get("previous_guesses", [])
        is_repeat = self.analyzer.is_repeat_guess(state["current_input"], previous_guesses)
        
        if is_repeat:
            state["similarity_score"] = 0
            logger.info("Repeated guess detected - no score change")
        else:
            # Build scoring prompt with all context
            scoring_prompt = self.templates.SCORING_SYSTEM.format(
                company_name=self.config.company_name,
                secret=self.config.secret_code,
                phase=state.get("phase", GamePhase.INITIAL.value),
                current_score=state.get("score", 0),
                input=state["current_input"],
                previous_guesses=", ".join(previous_guesses[-5:]) if previous_guesses else "None"
            )
            
            # Get scoring analysis
            score_result = self.llm_service.get_score(scoring_prompt)
            state["similarity_score"] = score_result["score_delta"]
            
            # If LLM detected correct answer, ensure we mark it as correct
            if score_result.get("is_correct_answer", False):
                state["is_correct"] = True
                # Calculate score_delta to reach 100
                current_score = state.get("score", 0)
                state["similarity_score"] = 100 - current_score
                logger.info(f"LLM detected correct answer - adjusting score to reach 100")
        
        # Add current input to previous guesses
        previous_guesses.append(state["current_input"])
        state["previous_guesses"] = previous_guesses[-20:]  # Keep last 20 guesses
        
        # Track discoveries
        new_discoveries = self.analyzer.detect_key_discoveries(
            state["current_input"],
            state,
            self.config
        )
        
        current_discoveries = state.get("discoveries", [])
        for discovery in new_discoveries:
            if discovery not in current_discoveries:
                current_discoveries.append(discovery)
                logger.info(f"New discovery: {discovery}")
        
        state["discoveries"] = current_discoveries
        
        return state
    
    def update_game_state(self, state: GameState) -> GameState:
        """Update score, phase, and determine if hints are needed"""
        # If already correct, skip score updates
        if state["is_correct"]:
            state["score"] = 100
            state["phase"] = GamePhase.COMPLETED.value
            return state
        
        # Update score
        current_score = state.get("score", 0)
        new_score = current_score + state["similarity_score"]
        
        # Cap score at 95 unless the secret code is found
        new_score = min(95, new_score)
        
        # Ensure score is between 0 and 100
        new_score = max(0, min(100, new_score))
        state["score"] = new_score
        
        # Update phase based on discoveries and score
        state["phase"] = self.analyzer.determine_phase(
            state.get("discoveries", []),
            new_score
        )
        
        # Check if hint is needed
        if state["is_hint_request"]:
            state["needs_hint"] = True
        elif new_score < 30 and state["attempts"] > 5:
            state["needs_hint"] = True
        else:
            state["needs_hint"] = False
        
        # Track wrong attempts for red herring detection
        if state["similarity_score"] < 0:
            wrong_attempts = state.get("wrong_attempts", [])
            wrong_attempts.append({
                "input": state["current_input"],
                "timestamp": datetime.now().isoformat()
            })
            state["wrong_attempts"] = wrong_attempts[-10:]  # Keep last 10
        
        logger.info(f"State updated - Score: {new_score}, Phase: {state['phase']}")
        return state
    
    def generate_response(self, state: GameState) -> GameState:
        """Generate contextual response based on game state"""
        # Build system prompt with current context
        system_prompt = self.templates.MAIN_SYSTEM.format(
            company_name=self.config.company_name,
            phase=state["phase"],
            score=state["score"],
            discoveries=", ".join(state.get("discoveries", [])) or "None yet",
            secret=self.config.secret_code
        )
        
        # Build message history
        messages = [SystemMessage(content=system_prompt)]
        
        # Add context about recent wrong attempts
        wrong_attempts = state.get("wrong_attempts", [])
        if wrong_attempts:
            last_wrong = wrong_attempts[-1]["input"]
            if "+" in last_wrong or "sum" in state["current_input"].lower():
                messages.append(SystemMessage(
                    content="User is adding numbers. Clarify it's a sequence, not a sum."
                ))
            elif any(date_indicator in last_wrong for date_indicator in ["2024", "april", "date"]):
                messages.append(SystemMessage(
                    content="User thinks it's a date. Clarify it's a sequence, not a date."
                ))
        
        # Check for repeated guesses
        if state["similarity_score"] == 0 and len(state.get("previous_guesses", [])) > 1:
            messages.append(SystemMessage(
                content="User repeated a previous guess. Acknowledge but don't give points."
            ))
        
        # Add the user's current input
        messages.append(HumanMessage(content=state["current_input"]))
        
        # Get response
        response = self.llm_service.get_response(messages)
        state["feedback"] = response
        
        # Add progress indicator only for wrong guesses with numbers
        if not state["is_correct"] and any(c.isdigit() for c in state["current_input"]) and state["similarity_score"] != 0:
            score = state["score"]
            if score >= 80:
                state["feedback"] += "\n[VERY CLOSE]"
            elif score >= 60:
                state["feedback"] += "\n[STRONG SIGNAL]"
            elif score >= 40:
                state["feedback"] += "\n[SIGNAL DETECTED]"
            elif score > 0:
                state["feedback"] += "\n[WEAK SIGNAL]"
        
        return state
    
    def provide_hint(self, state: GameState) -> GameState:
        """Provide contextual hints based on progress"""
        score = state.get("score", 0)
        hints_revealed = state.get("hints_revealed", [])
        
        # Determine which hint to show
        if score < 30 and "hint_1" not in hints_revealed:
            hint = self.templates.HINT_TEMPLATES["hint_1"]
            hints_revealed.append("hint_1")
        elif score < 50 and "hint_2" not in hints_revealed:
            hint = self.templates.HINT_TEMPLATES["hint_2"]
            hints_revealed.append("hint_2")
        elif score < 70 and "hint_3" not in hints_revealed:
            hint = self.templates.HINT_TEMPLATES["hint_3"]
            hints_revealed.append("hint_3")
        else:
            hint = "💭 'You have all the pieces... D-O-X... positions in the alphabet... read them in order.'"
        
        state["hints_revealed"] = hints_revealed
        state["feedback"] = f"**[ENCRYPTED HINT RECEIVED]**\n\n{hint}"
        
        return state
    
    def handle_victory(self, state: GameState) -> GameState:
        """Handle successful code entry"""
        state["phase"] = GamePhase.COMPLETED.value
        state["score"] = 100  # Ensure score is 100 for victory
        state["end_time"] = datetime.now().isoformat()
        
        victory_message = """
🎯 **ACCESS GRANTED**

The terminal flickers to life. You're in.

*"Excellent work. The server is unlocked. I'm downloading the evidence now..."*

Files cascade across your screen:
- SECRET_MERGER_PROPOSAL.pdf
- BOARD_COMMUNICATIONS.enc
- PROJECT_SHADOWNET.docs

*"My God... it's bigger than we thought. They're planning to merge with a competitor and lay off thousands while executives cash out. This evidence will stop them."*

*"Thank you, whoever you are. You've saved countless jobs and exposed the truth. The whistleblower network will remember this."*

**[MISSION COMPLETE]**
Code: {code}
Attempts: {attempts}
Score: {score}/100
"""
        
        state["feedback"] = victory_message.format(
            code=self.config.secret_code,
            attempts=state.get("attempts", 0),
            score=state.get("score", 100)
        )
        
        logger.info(f"Game completed - Attempts: {state['attempts']}, Score: {state['score']}")
        return state
    
    def create_initial_state(self, session_id: Optional[str] = None) -> GameState:
        """Create a fresh game state"""
        return {
            "messages": [],
            "phase": GamePhase.INITIAL.value,
            "score": 0,
            "attempts": 0,
            "current_input": "",
            "feedback": "",
            "discoveries": [],
            "hints_revealed": [],
            "wrong_attempts": [],
            "previous_guesses": [],  # Track all previous guesses
            "session_id": session_id or datetime.now().isoformat(),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "similarity_score": 0,
            "is_hint_request": False,
            "needs_hint": False,
            "is_correct": False
        }
    
    def process_turn(self, user_input: str, state: Optional[GameState] = None) -> Dict[str, Any]:
        """Process a single game turn - main integration point"""
        # Initialize state if not provided
        if state is None:
            state = self.create_initial_state()
        
        # Set current input
        state["current_input"] = user_input
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Return integration-friendly response
        return {
            "response": result["feedback"],
            "state": result,
            "metadata": {
                "score": result["score"],
                "phase": result["phase"],
                "attempts": result["attempts"],
                "completed": result["phase"] == GamePhase.COMPLETED.value
            }
        }

# ============= Public API =============
class GameAPI:
    """Clean API for integration with FastAPI/Streamlit"""
    
    def __init__(self, config: Optional[GameConfig] = None):
        self.game = AmdocsConspiracyGame(config)
        self.sessions: Dict[str, GameState] = {}
    
    def start_new_game(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Start a new game session"""
        session_id = session_id or datetime.now().isoformat()
        initial_state = self.game.create_initial_state(session_id)
        self.sessions[session_id] = initial_state
        
        opening_message = """
**[ENCRYPTED MESSAGE RECEIVED]**

*"I need your help. There's something big at Amdocs, but the server's locked. The code's tied to the company's market identity."*

—Anonymous Whistleblower

**[CONNECTION ESTABLISHED]**

Type your questions to uncover the truth. The clock is ticking...
"""
        
        return {
            "session_id": session_id,
            "message": opening_message,
            "state": "active"
        }
    
    def process_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process a user message in a game session"""
        if session_id not in self.sessions:
            return {
                "error": "Session not found. Please start a new game.",
                "state": "error"
            }
        
        # Get current state
        state = self.sessions[session_id]
        
        # Process the turn
        result = self.game.process_turn(message, state)
        
        # Update session state
        self.sessions[session_id] = result["state"]
        
        return {
            "session_id": session_id,
            "message": result["response"],
            "metadata": result["metadata"],
            "state": "completed" if result["metadata"]["completed"] else "active"
        }
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a game session"""
        if session_id not in self.sessions:
            return {"error": "Session not found"}
        
        state = self.sessions[session_id]
        return {
            "session_id": session_id,
            "phase": state["phase"],
            "score": state["score"],
            "attempts": state["attempts"],
            "discoveries": state["discoveries"],
            "started": state["start_time"],
            "completed": state.get("end_time") is not None
        }

# ============= CLI Interface =============
def run_cli_game():
    """Run the game in CLI mode for testing"""
    print("\n" + "="*60)
    print("CORPORATE CONSPIRACY: THE AMDOCS CODE")
    print("="*60)
    
    api = GameAPI()
    start_response = api.start_new_game()
    session_id = start_response["session_id"]
    
    print("\n" + start_response["message"])
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\n[CONNECTION TERMINATED]")
            break
        
        response = api.process_message(session_id, user_input)
        print("\n" + response["message"])
        
        if response["state"] == "completed":
            break

# if __name__ == "__main__":
#     # run_cli_game()
#     AmdocsConspiracyGame()._build_graph()