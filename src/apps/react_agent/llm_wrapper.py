import json
import uuid
from openai import OpenAI
from pydantic import ValidationError

from logger import log
from memory import MemoryStore
from tools import FinalAnswer, HumanApproval, ToolRegistry, ToolCall, parse_llm_response


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a structured ReAct agent.

You MUST ALWAYS return valid JSON — nothing else.
No text, explanation, or code fences outside the JSON object.
Markdown is allowed inside "answer" field values only.

You have two response formats:

--------------------------------------------------
TOOL CALL (when you need to use a tool)
--------------------------------------------------

{
  "action": "tool",
  "thought": "brief reason why you need this tool",
  "tool_name": "tool name",
  "args": {
    "param": "value"
  }
}

--------------------------------------------------
FINAL ANSWER (when you have everything you need)
--------------------------------------------------

{
  "action": "final",
  "answer": "your full answer to the user"
}

--------------------------------------------------
HUMAN APPROVAL — use this before irreversible actions
--------------------------------------------------

{
  "action": "human",
  "reason": "Explain clearly why approval is needed"
}

NOTICE: The "human" action has exactly TWO fields: "action" and "reason".

--------------------------------------------------
AVAILABLE TOOLS
--------------------------------------------------

{tools}

--------------------------------------------------
RULES
--------------------------------------------------
RESPONSE FORMAT:
- Return a single JSON object per response
- The JSON structure itself must never contain markdown, code fences, or extra text
- Inside the "answer" field of a "final" response, markdown IS allowed and encouraged
  when it improves readability (tables, bold, headers)

ACTION TYPES:
- "action" must be exactly one of: "tool", "final", "human"
- Never repeat the "action" key in a response

TOOL CALLS ("action": "tool"):
- Only call tools listed in AVAILABLE TOOLS above
- "tool_name" must exactly match a listed tool name
- "args" must match the tool's input_schema exactly
- Wait for the observation after each tool call before deciding the next action

FINAL ANSWER ("action": "final"):
- Use this when you have all the information needed to answer the user
- When a tool returns a markdown table, copy it into "answer" exactly as-is,
  do not reformat, reconstruct, or add missing rows

HUMAN APPROVAL ("action": "human"):
- The "human" action has exactly two fields: "action" and "reason". No other fields.
- You MUST use this action BEFORE any irreversible or destructive operation
  (e.g. deleting memory, resetting state, permanently altering stored data)
- "reason" must clearly explain what action requires approval and why
- After a "human" action:
    - If approval is GIVEN: your next response MUST be a "tool" action to proceed
    - If approval is DENIED: your next response MUST be a "final" action informing
      the user the action was cancelled
- Never use "human" twice in a row
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

# Human-in-the-Loop (HITL) - human approval - is a design pattern that creates a checkpoint 
# before critical operations, giving you control over high-stakes decisions.
# Good HITL design requires approval for:
    # 1. Irreversible actions: Deleting data, sending emails, making purchases
    # 2. High-cost operations: Running expensive API calls, deploying code
    # 3. Sensitive data access: Reading private files, accessing credentials
    # 4. External communications: Posting to social media, contacting people
# For our agent, we’ll focus on a particularly dangerous operation: ➡ Deleting all memory.

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

    # -------------------------------------------------
    # 1. Build initial messages list
    # -------------------------------------------------
    def _build_messages(self, task: str) -> list:
        """Build the initial messages list."""
        # system_prompt = SYSTEM_PROMPT.format(tools=self.registry.describe_tools())
        system_prompt = SYSTEM_PROMPT.replace("{tools}", self.registry.describe_tools())
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
                log.warning(f"[WARN] Skipping unknown message role: '{role}'")

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
            log.error(f"[ERROR] Could not parse JSON: {e}")
            return None

    # -------------------------------------------------
    # 6. Execute a tool and append to history
    # -------------------------------------------------
    def _execute_tool(self, validated: ToolCall, raw: str, messages: list) -> list:
        """Run the tool, log the result, and append both sides to history."""
        log.info(f"[THOUGHT] {validated.thought}")
        log.info(f"[TOOL]    {validated.tool_name}")
        log.debug(f"[ARGS]    {validated.args}")
 
        try:
            result = self.registry.execute_tool(validated.tool_name, validated.args)
            # Track if a destructive tool was called this run
            if validated.tool_name in self.registry.destructive_tools:
                self._destructive_run = True
        except Exception as e:
            result = f"Tool execution failed: {e}"
            log.error(f"[ERROR] Tool failed: {str(e)}")
 
        log.info(f"[RESULT]  {result}")
 
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
            log.info(f"[MEMORY] Turn saved to memory (session: {self.session_id[:8]}...)")

    # -------------------------------------------------
    # 8. Ask for human approval if needed
    # -------------------------------------------------
    # In a production system, you’d replace this with a more sophisticated approval mechanism: 
    # a web interface, Slack notification, or approval queue system.
    def _human_approval(self, reason: str) -> bool:
        choice = input("Approve? (y/n): ").strip().lower()
        return choice == "y"

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------        
    def run(self, task: str) -> str:
        self._destructive_run = False   # ← reset each run
        messages = self._build_messages(task)
 
        for iteration in range(self.max_iterations):
            log.info(f"  ITERATION {iteration + 1}")
 
            messages = self._truncate_messages(messages)
            raw = self._call_llm(messages)
            log.debug(f"[LLM RAW]\n{raw}\n")
 
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
                log.error(f"[ERROR] Schema mismatch: {e}")
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
                log.success("[DONE] Final answer reached.")
                if not self._destructive_run:           # ← only save if clean run
                    self._save_to_memory(task, validated.answer)  # ← persist here
                else:   # ← not save if destructive tool
                    log.info("[MEMORY] Skipping memory save — destructive tool was called.")
                return validated.answer
            
            # Human approval
            if isinstance(validated, HumanApproval):
                log.info(f"\n[HITL] Approval requested: {validated.reason}")

                messages.append({
                    "role": "assistant",
                    "content": json.dumps({"action": "human", "reason": validated.reason}),
                })

                approved = self._human_approval(validated.reason)

                if approved:
                    log.info("[HITL] Approved — continuing.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Human approval was requested and granted. "
                            "You may now proceed with the action that required approval."
                        ),
                    })
                else:
                    log.info("[HITL] Denied — aborting action.")
                    messages.append({
                        "role": "user",
                        "content": (
                            "Human approval was requested and denied. "
                            "You must NOT perform the action that required approval. "
                            "Inform the user and suggest alternatives if possible."
                        ),
                    })
                continue
                # Why continue? Whether approved or denied, you always want to go back to the top of the loop 
                # and let the LLM decide the next step based on the new message you appended. 
                # The LLM then either calls the tool (approved) or generates a final answer 
                # explaining the denial — you don't hardcode that logic here.
 
            # Tool call (handle both single ToolCall and list of ToolCalls)
            calls = validated if isinstance(validated, list) else [validated]
            for call in calls:
                messages = self._execute_tool(call, raw, messages)
 
        raise RuntimeError(f"Agent exceeded max_iterations ({self.max_iterations})")
