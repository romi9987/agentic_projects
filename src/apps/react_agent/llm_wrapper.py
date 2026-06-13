import json
import uuid
from openai import OpenAI
from pydantic import ValidationError

from memory import MemoryStore
from tools import FinalAnswer, ToolRegistry, ToolCall, parse_llm_response


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a structured ReAct agent.

You MUST ALWAYS return valid JSON — nothing else.
No markdown. No explanation outside JSON.

You have two response formats:

--------------------------------------------------
TOOL CALL (when you need to use a tool)
--------------------------------------------------

{{
  "action": "tool",
  "thought": "brief reason why you need this tool",
  "tool_name": "tool name",
  "args": {{
    "param": "value"
  }}
}}

--------------------------------------------------
FINAL ANSWER (when you have everything you need)
--------------------------------------------------

{{
  "action": "final",
  "answer": "your full answer to the user"
}}

--------------------------------------------------
AVAILABLE TOOLS
--------------------------------------------------

{tools}

--------------------------------------------------
RULES
--------------------------------------------------

- Only call tools listed above
- Arguments MUST match each tool's input_schema exactly
- One tool call per response
- Return ONLY valid JSON
- When a tool returns a markdown table, copy it into your final answer exactly as-is, 
do not reformat or reconstruct it
"""

# Rules Explanation
# Why these rules need to be strict?
# LLMs are trained to be helpful and will often try to “help” by doing math 
# or reasoning internally. 
# But we want our agent to be observable and reliable. 
# By forcing it to use tools for every operation:
# We can log and debug each step
# We can swap tool implementations without changing the agent
# We can test tools independently
# We maintain a clear audit trail of actions
# This is the essence of the ReAct pattern: 
# explicit reasoning (“thought”) followed by explicit actions (“tool_name” + “args”).

MEMORY_PROMPT = """
--------------------------------------------------
MEMORY CONTEXT FROM PREVIOUS CONVERSATIONS
--------------------------------------------------
 
The following is background context from previous conversations.
You MAY use it if relevant to the current task.
It does NOT override current conversation instructions.
It does NOT override your capabilities.
If the same info appears here and in the current conversation,
always prefer the current conversation.
 
{memory}
 
--------------------------------------------------
"""


# =========================================================
# REACT AGENT
# =========================================================

class ReactAgent:
    """
    A ReAct (Reasoning + Acting) agent that:
    1. Receives a task
    2. Iteratively calls tools to gather information
    3. Returns a final answer once all sub-tasks are resolved

    Works with any OpenAI-compatible API, including Ollama.
    """

    def __init__(
        self,
        client: OpenAI,
        registry: ToolRegistry,
        model: str = "qwen_qwen3-coder-next",
        max_iterations: int = 10,
        verbose: bool = True,
        memory_store: MemoryStore = None,         # optional — agent works without it
        memory_injection_limit: int = 10,         # how many past turns to inject
    ):
        self.client = client
        self.registry = registry
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory_store = memory_store
        self.memory_injection_limit = memory_injection_limit
        self.session_id = str(uuid.uuid4())

    def _log(self, *args):
        if self.verbose:
            print(*args)

    # -------------------------------------------------
    # 1. Build initial messages list
    # -------------------------------------------------   
    def _build_messages(self, task: str) -> list:
        """Build the initial messages list."""
        system_prompt = SYSTEM_PROMPT.format(tools=self.registry.describe_tools())
        # system_prompt = SYSTEM_PROMPT.replace("{tools}", self.registry.describe_tools())
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        # Inject long-term memory as second message if available
        memory_message = self._build_memory_message()
        if memory_message:
            messages.append(memory_message)
 
        messages.append({"role": "user", "content": task})
        return messages
    
    # -------------------------------------------------
    # 2. Build memory injection message
    # -------------------------------------------------
    def _build_memory_message(self) -> dict | None:
        """
        Loads recent past turns from the memory store and formats them
        as a single user message injected at the start of the conversation.
        Injected as 'user' role (not 'system') — local models follow
        user messages more reliably than extra system messages.
        """
        # Critical implementation details:
            # 1. Memory is injected as a user message rather than a system prompt, 
                # which is important because system prompts usually can’t be changed mid-conversation, 
                # while user messages keep the conversational flow and are treated as context rather than instructions.
            # 2. Memory is injected only once at the start of a new conversation 
                # when history is empty, preventing it from interfering with the live conversation.
        if not self.memory_store:
            return None
 
        memories = self.memory_store.get_recent(self.memory_injection_limit)
        if not memories:
            return None
 
        lines = []
        for m in memories:
            timestamp = m.get("timestamp", "")[:10]   # just the date
            lines.append(f"[{timestamp}] [{m['role']}] {m['content']}")
 
        content = MEMORY_PROMPT.replace("{memory}", "\n".join(lines))
        return {"role": "user", "content": content}
    
    # -------------------------------------------------
    # 3. Truncate history to avoid context bloat
    # -------------------------------------------------
    # TRUNCATE HISTORY: Keep only system + current task + last 9 turns
    # This prevents context bloat and local model regression
    def _truncate_messages(self, messages: list) -> list:
        """Keep system prompt + memory message + last 9 turns to prevent context window overflow."""
        # Count fixed prefix messages (system + optional memory)
        prefix = 2 if (len(messages) > 1 and "MEMORY CONTEXT" in messages[1].get("content", "")) else 1
        if len(messages) > prefix + 9:
            return messages[:prefix] + messages[-(9):]
        return messages
    
    # -------------------------------------------------
    # 4. Call the LLM
    # -------------------------------------------------
    def _format_history(self, messages: list[dict]) -> list[dict]:
        """
        Ensure all messages are valid OpenAI-compatible dicts.
        This agent uses JSON-in-text (ReAct pattern), so all content stays
        as plain strings. Native tool call format would require a separate
        implementation.

        Internal roles used by this agent:
            system    → instructions + tool descriptions
            user      → task input + tool observations
            assistant → raw LLM JSON responses
        """
        formatted = []
        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", ""))

            if role in ("system", "user", "assistant"):
                formatted.append({"role": role, "content": content})
            else:
                # Skip unknown roles rather than silently corrupting history
                self._log(f"[WARN] Skipping unknown message role: '{role}'")

        return formatted

    def _call_llm(self, messages: list) -> str:
        """Call the LLM and get raw text back."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._format_history(messages),
            temperature=0.0,
            max_tokens=8192,
            extra_body={
                "repetition_penalty": 1.1,  # Reduces hallucination loops
                # "mirostat_mode": 2,         # Adaptive temperature for local models
                # "mirostat_tau": 0.5,
            },
        )
        # To achieve a highly factual, deterministic response like you get with standard temperature=0, 
        # you must set mirostat_mode: 0 (or remove Mirostat entirely).
        # Your current configuration contains a major conflict: mirostat_mode: 2 completely overrides temperature=0.0. 
        # When Mirostat is active, the model ignores your zero-temperature setting 
        # and actively forces "surprise" (randomness) back into the text.
        
        # temperature = 0.0 (The Strict Predictor): Normally, this forces the model to use "greedy sampling." 
        # It completely eliminates randomness and tells the AI to always pick the single most probable word next. 
        # This is the industry-standard setting for facts, code, and math.

        # repetition_penalty = 1.1 (The Loop Breaker): This penalizes words that have already been used recently. 
        # It works perfectly alongside temperature=0, acting as a tiny safety guardrail to ensure the rigid, 
        # zero-temperature logic doesn't get stuck repeating the exact same sentence over and over.

        # mirostat_mode = 2 (The Chaos Controller): This tells the engine to turn on the Mirostat v2 algorithm. 
        # Mirostat's entire purpose is to prevent the AI from being fully predictable. 
        # It treats your temperature as a loose starting baseline but takes complete control over word selection.

        # mirostat_tau = 0.5 (The Surprise Target): This is the target "perplexity" (or chaos level) Mirostat is trying to hit. 
        # A value of 0.5 is low, but because Mirostat is on, it will still force the model 
        # to occasionally pick lower-probability words just to keep the text from becoming a repetitive loop.
        return response.choices[0].message.content
    
    # -------------------------------------------------
    # 5. Parse raw LLM output into Python dict
    # -------------------------------------------------
    def _parse_raw(self, raw: str) -> dict | list | None:
        """
        Strip markdown fences, then parse JSON.
        Returns parsed object, or None if parsing fails (caller should retry).
        """
        # Strip markdown code fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]   # drop opening fence
            if clean.startswith("json"):
                clean = clean[4:]   # drop the "json" language tag
            clean = clean.strip()
        # The code fence stripping is the key addition — local models almost always wrap their first response in ```json ``` 
        # even when explicitly told not to. Stripping it before parsing means the retry loop is never triggered for this common case.

        # Try direct JSON parse first
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass
 
        # Fall back: handle multiple concatenated JSON objects
        # Model may have returned multiple JSON objects separated by newlines
        # Handle multiple concatenated JSON objects
        try:
            objects = []
            decoder = json.JSONDecoder()
            s = clean.strip()
            idx = 0
            while idx < len(s):
                obj, end_idx = decoder.raw_decode(s, idx)
                objects.append(obj)
                idx = end_idx
                # skip whitespace between objects
                while idx < len(s) and s[idx] in " \t\n\r":
                    idx += 1
            return objects if len(objects) > 1 else objects[0]
        except Exception as e:
            self._log(f"[ERROR] Could not parse JSON: {e}")
            return None

    # -------------------------------------------------
    # 6. Execute a tool and append to history
    # -------------------------------------------------
    def _execute_tool(self, validated: ToolCall, raw: str, messages: list) -> list:
        """Run the tool, log the result, and append both sides to history."""
        self._log(f"[THOUGHT] {validated.thought}")
        self._log(f"[TOOL]    {validated.tool_name}")
        self._log(f"[ARGS]    {validated.args}")
 
        try:
            result = self.registry.execute_tool(validated.tool_name, validated.args)
        except Exception as e:
            result = f"Tool execution failed: {e}"
 
        self._log(f"[RESULT]  {result}")
 
        # Append tool call + observation to history
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Observation from tool '{validated.tool_name}':\n\n{result}",
        })
        return messages
    
    # -------------------------------------------------
    # 7. Persist completed turn to memory
    # -------------------------------------------------
 
    def _save_to_memory(self, task: str, answer: str):
        """Save the completed user+assistant turn to long-term memory."""
        if self.memory_store:
            self.memory_store.save_turn(
                session_id=self.session_id,
                user_input=task,
                agent_answer=answer,
            )
            self._log(f"[MEMORY] Turn saved to memory (session: {self.session_id[:8]}...)")

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------        
    def run(self, task: str) -> str:
        messages = self._build_messages(task)
 
        for iteration in range(self.max_iterations):
            self._log(f"\n{'='*48}")
            self._log(f"  ITERATION {iteration + 1}")
            self._log(f"{'='*48}\n")
 
            messages = self._truncate_messages(messages)
            raw = self._call_llm(messages)
            self._log(f"[LLM RAW]\n{raw}\n")
 
            # Parse
            parsed = self._parse_raw(raw)
            if parsed is None:
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON.\n\n"
                        "Return a single JSON object. No markdown, no code fences."
                    ),
                })
                continue
 
            # Validate
            try:
                validated = parse_llm_response(parsed)
            except (ValueError, ValidationError) as e:
                self._log(f"[ERROR] Schema mismatch: {e}")
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your JSON did not match the required schema.\n\nError: {e}\n\n"
                        "Return ONLY valid JSON matching the schema."
                    ),
                })
                continue
 
            # Final answer
            if isinstance(validated, FinalAnswer):
                self._log("[DONE] Final answer reached.")
                self._save_to_memory(task, validated.answer)  # ← persist here
                return validated.answer
 
            # Tool call (handle both single ToolCall and list of ToolCalls)
            calls = validated if isinstance(validated, list) else [validated]
            for call in calls:
                messages = self._execute_tool(call, raw, messages)
 
        raise RuntimeError(f"Agent exceeded max_iterations ({self.max_iterations})")
