from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AgentState:
    user_request: str
    context: Dict[str, Any] = field(default_factory=dict)
    thoughts: List[str] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    done: bool = False
    

    def __init__(self, user_request: str):
        self.user_request = user_request
        self.thoughts = []
        self.observations = []
        self.errors = []
        self.done = False

        # NEW FIELDS
        self.flight_args = None
        self.hotel_args = None
        self.preferences = None

    def add_thought(self, thought: str):
        self.thoughts.append(thought)

    def add_observation(self, obs: Dict[str, Any]):
        self.observations.append(obs)

    def add_error(self, err: str):
        self.errors.append(err)
