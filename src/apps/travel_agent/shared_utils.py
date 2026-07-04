import asyncio
import json
import time
from typing import Any, Dict, Optional, Tuple, Awaitable, Callable, List

from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI

from observer import write_jsonl

# LM Studio / OpenAI-compatible client
client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")


def validate_tool_args(
    schema: type[BaseModel],
    raw_args: Optional[Dict[str, Any]],
) -> Tuple[Optional[BaseModel], Optional[str]]:
    """
    Validate raw LLM arguments using a Pydantic model.
    Returns (validated_model, None) on success,
    or (None, error_message) on failure.
    """
    if raw_args is None:
        return None, "no arguments provided"

    try:
        validated = schema(**raw_args)
        return validated, None
    except ValidationError as e:
        errors = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in e.errors()
        )
        return None, errors


async def repair_tool_args(client, tool_name: str, args: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
    prompt = f"""
Tool: {tool_name}
Original args (JSON): {json.dumps(args, default=str)}
Validation errors: {errors}

Return ONLY fixed JSON args, no explanation.
"""

    resp = await client.chat.completions.create(
        model="qwen_qwen3-coder-next",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content

    try:
        fixed = json.loads(raw)
    except json.JSONDecodeError:
        write_jsonl({"event": "repair.invalid_json", "tool": tool_name, "raw": raw})
        return args  # fallback

    write_jsonl({"event": "repair.success", "tool": tool_name, "fixed": fixed})
    return fixed


async def execute_tool(
    tool_name: str,
    schema: type[BaseModel],
    raw_args: Optional[Dict[str, Any]],
    api_fn: Callable[[BaseModel], Awaitable[list[Dict[str, Any]]]],
) -> Tuple[Optional[list[Dict[str, Any]]], list[str], list[Dict[str, Any]]]:
    """
    3-layer defense for tool execution:
    1. Validate args with Pydantic
    2. If invalid → one-shot LLM repair
    3. Execute tool with validated args

    Returns: (results, errors, debug_logs)
    """
    if not raw_args:
        return None, [], []

    t0 = time.perf_counter()
    debug: Dict[str, Any] = {"tool": tool_name, "raw_args": raw_args}
    errors: list[str] = []

    # --- Layer 1: Validate ---
    validated, err = validate_tool_args(schema, raw_args)

    if err:
        debug["validation_error"] = err

        # --- Layer 2: Repair ---
        repaired = await repair_tool_args(tool_name, raw_args, err, schema)

        if repaired is None:
            debug.update({"repair": "failed", "status": "skipped"})
            debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
            errors.append(f"{tool_name}: repair failed — {err}")
            return None, errors, [debug]

        validated = repaired
        debug.update({
            "repair": "succeeded",
            "repaired_args": validated.model_dump(),
        })

    # --- Layer 3: Execute tool ---
    try:
        results = await api_fn(validated)
        debug.update({
            "status": "ok",
            "final_args": validated.model_dump(),
            "result_count": len(results),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        })
        return results, errors, [debug]

    except Exception as e:
        debug.update({"status": "api_error", "error": str(e)})
        debug["elapsed_ms"] = round((time.perf_counter() - t0) * 1000)
        errors.append(f"{tool_name}: API error — {e}")
        return None, errors, [debug]


async def retry_node(fn, state, *, retries=2, delay=0.5, name="node"):
    # With these pieces in place, your pipeline now has:
        # Pydantic validation at each LLM boundary and tool call
        # Per‑node retries with backoff
        # Structured error propagation into BookingState.errors
        # Logfire spans and logs for every attempt and failure
    """
    Retry a pipeline node on errors/validation failures.
    Node must return a dict; if it contains "errors", it's treated as failed.
    """
    attempt = 0

    while attempt <= retries:
        write_jsonl({"event": "retry.attempt", "node": name, "attempt": attempt})

        try:
            result = await fn(state)

            if isinstance(result, dict) and result.get("errors"):
                write_jsonl({
                    "event": "retry.validation_error",
                    "node": name,
                    "attempt": attempt,
                    "errors": result["errors"],
                })
                raise ValueError("validation failed")

            return result

        except Exception as e:
            write_jsonl({
                "event": "retry.error",
                "node": name,
                "attempt": attempt,
                "error": str(e),
            })

            if attempt == retries:
                return {
                    "errors": [f"{name} failed after {retries+1} attempts: {e}"],
                    "failed": True,
                }

            await asyncio.sleep(delay)
            attempt += 1
