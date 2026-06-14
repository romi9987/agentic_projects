"""
Logger setup for the ReAct agent using loguru.

Log levels — what appears where:

    Event                    Level     Console        File
    -------------------------------------------------------
    Raw LLM response         DEBUG     verbose only   no
    Tool args                DEBUG     verbose only   no
    Tool name + result       INFO      yes            yes
    Task started             INFO      yes            yes
    Final answer             SUCCESS   yes            yes
    Memory saved/skipped     INFO      yes            yes
    HITL approval            INFO      yes            yes
    Memory deletion          WARNING   yes            yes
    JSON parse error         ERROR     yes            yes

Console shows DEBUG+ when verbose=True, WARNING+ when verbose=False.
File always writes INFO and above, rotates at 10MB, retains for 7 days.

The idea is that DEBUG is noise you want when actively developing, 
INFO and above is what you want in the file as a permanent audit trail — 
tool calls, results, memory operations, HITL decisions. 
Raw LLM output is too verbose for the file but useful on the console while debugging.
"""

import os
import sys
from loguru import logger


log_file = os.getenv("AGENT_LOG_PATH")

def setup_logger(log_file: str = log_file, verbose: bool = True) -> None:
    """
    Configure loguru for the agent.
    - Console: shows everything if verbose, warnings+ if not
    - File: always writes INFO and above, rotates at 10MB, keeps 7 days
    """
    logger.remove()  # remove loguru's default handler

    # Console handler
    logger.add(
        sys.stdout,
        level="DEBUG" if verbose else "WARNING",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )

    # File handler — only the important stuff
    logger.add(
        log_file,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
        rotation="10 MB",    # new file after 10MB
        retention="7 days",  # delete logs older than 7 days
        compression="zip",   # compress rotated files
        encoding="utf-8",
    )


# Single shared instance — import this everywhere
log = logger
