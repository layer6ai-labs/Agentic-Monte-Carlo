import re

NO_ACTION = "[no action]"

class ActionParserError(Exception):
    pass

def parse_action(llm_response: str) -> str:
    # 1. Pre-cleaning: Remove LLM-specific artifacts (From Snippet 2)
    # This prevents invisible tokens from breaking the regex or the server.
    clean_response = llm_response.replace("</s>", "").replace("<|eot_id|>", "")

    # 2. Regex Extraction (Combined)
    # - re.IGNORECASE: Handles "action:", "ACTION:", "Action:" (From Snippet 2)
    # - re.DOTALL: Allows us to parse multi-line outputs correctly if needed.
    # - (?=\n|$): Lookahead to stop strictly at a newline or end of text (From Snippet 1)
    matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", clean_response, re.IGNORECASE | re.DOTALL)

    # 3. Multi-Action Safety Check (From Snippet 1)
    if len(matches) > 1:
        raise ActionParserError("Error: Multiple actions detected. Please output only one 'Action:'.")

    if not matches:
        raise ActionParserError("Error: No 'Action:' keyword found in response.")

    raw_action = matches[0]

    # 4. Aggressive Sanitization (From Snippet 1)
    # Whitelisting is safer than blacklisting. We allow letters, numbers, spaces, and commas.
    # We strip punctuation like '.' or '!' which confuse game parsers.
    sanitized_action = re.sub(r"[^A-Za-z0-9, ]+", "", raw_action)

    # 5. Normalization (From Snippet 1)
    # Collapses "craft   plank" -> "craft plank"
    final_action = " ".join(sanitized_action.split()).strip()

    if not final_action:
        raise ActionParserError("Error: Action field was empty.")

    return final_action