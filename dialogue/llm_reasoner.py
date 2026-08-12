# dialogue/llm_reasoner.py
import json
import re
from dialogue.llm_interface import query_llm


def _extract_json(text: str):
    # Reliably extract the first complete JSON object from the output of a large language model.
    if not text:
        return None
    # Remove the Markdown code block
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def reason_about_input(user_input, context):
    history_text = ""
    for item in context["history"]:
        history_text += f"{item['speaker']}: {item['text']}\n"

    prompt = f"""You are an advanced interactive robot reasoning layer. Analyze the user input based on the live environment.

[DYNAMIC ENVIRONMENT CONTEXT]
The robot can pick up exactly ONE object from the table. The objects currently on the table, with their perceived attributes (name, color, shape, table region, relative position), are:
{context['objects_manifest_text']}
Current Robot Target Status: {context['current_target']}
Dialogue History:
{history_text}

[USER INPUT TO PROCESS]
"{user_input}"

[TASK RULES]
1. "is_pure_query" = TRUE only when the user is NOT commanding a real action now: e.g. a hypothetical ("if you had to choose..."), asking which/what/why, or asking about attributes/scene. In that case do NOT change the target.
2. "is_pure_query" = FALSE when the user commands or delegates an action now (pick/select/change/"choose one for me").
3. Resolve descriptions to a concrete object: if the user refers to an object by an attribute (color, shape, table region, relative position) rather than its exact name (e.g. "the green one", "the round fruit", "the one on the left"), map it to the matching EXACT object name from the context and put that EXACT name in "object_name".
4. "intent":
   - "select"          -> user wants a specific object (named or described).
   - "reject"          -> user refuses the current target / wants the other one.
   - "delegate_choice" -> user lets you decide which to pick.
   - "ask_why"/"ask_scene"/"ask_current" -> questions (pure query).
   - "unknown" if unclear.
5. "object_name" must be one of the exact names above, or null if the user did not single one out.
6. Ground every spatial/attribute claim in the data above — do NOT invent or flip positions. When describing where something is, use its given table region and relative_position exactly as listed.
7. If the user names or describes an object that is NOT present above (e.g. an item that isn't on the table), do NOT guess a different object: set "object_name" to null, keep intent "select", and in "explanation" tell them it isn't on the table and ask which of the listed objects they want.
8. Match colors and shapes LOOSELY by shade/family, never literally. A color word matches any object whose listed color is a shade or compound of it: "green" matches "yellow-green", "olive green", or "dark green"; "red" matches "dark red", "pink-red", or "dark pink-red"; "yellow" matches "yellow-green". Treat shape synonyms too ("long"≈"elongated", "circular"/"ball"≈"round"). NEVER claim a color or shape is absent when some object's listed attribute is a shade, compound, or synonym of it — resolve to that object instead. Only leave object_name null and ask if two or more objects match the description equally well.

[OUTPUT FORMAT]
Return a SINGLE JSON object. No markdown, no extra text.
{{
    "reasoning": "Briefly explain the classification.",
    "is_pure_query": true/false,
    "intent": "select/reject/confirm/ask_current/ask_scene/ask_why/delegate_choice/unknown",
    "object_name": "exact object name if singled out, else null",
    "explanation": "Polite natural-language reply to show the user."
}}

Response JSON:"""

    response = query_llm(prompt)
    print("\n--- OPEN-WORLD REASONING LOG ---")
    print(response)
    print("--------------------------------")

    parsed = _extract_json(response)
    if parsed is not None:
        return {
            "is_pure_query": parsed.get("is_pure_query", False),
            "intent": parsed.get("intent", "unknown"),
            "object_name": parsed.get("object_name", None),
            "explanation": parsed.get("explanation", ""),
        }
    else:
        print("[Error Parsing JSON]: no balanced JSON object found in response")

    return {"is_pure_query": False, "intent": "unknown", "object_name": None, "explanation": ""}