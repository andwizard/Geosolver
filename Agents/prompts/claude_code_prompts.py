CLAUDE_CODE_PROMPT = """

4.You MUST use the StructuredOutput tool to submit your final result as a JSON object with these fields:
- "response": "ACCEPT" if you successfully produced a valid script, "FAILED" otherwise
- "script": The complete, runnable Python script content (empty string if FAILED)
- "answer": A textual summary of the results/outputs produced by the script
5. If you cannot produce a valid script, set "response" to "FAILED" and "script" to ""
6. Return ONLY the JSON object via StructuredOutput — no extra explanation, no markdown fences
"""
