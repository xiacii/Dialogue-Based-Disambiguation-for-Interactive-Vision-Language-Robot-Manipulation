# dialogue_manager.py
from dialogue.llm_reasoner import reason_about_input

class DialogueManager:
    def __init__(self, state_manager):
        self.state = state_manager

    def process(self, user_input):
        context = self.state.get_context()
        objects = context["objects"]
        current_target = context["current_target"]
        pending_proposal = context["pending_proposal"]
        
        user_raw = user_input.strip()
        user_lower = user_raw.lower()

        # A default safety net, maintaining the status quo at the hardware signal level
        output = {
            "user_response": "",
            "robot_signal": {
                "target": current_target, 
                "action_modifier": "keep"
            }
        }

        # Fast Track Layer: Ultra-low-latency circuit breaker, handling full-word matches
        is_positive_confirmation = user_lower in ["ok", "yes", "sure", "correct", "fine", "accept", "do it"]
        if is_positive_confirmation:
            if pending_proposal: # Directly lock the proposal that was put on hold in the previous round by the agent
                self.state.update_target(pending_proposal)
                output["user_response"] = f"Agent: Dynamic proposal accepted. Targeting the {pending_proposal}."
                output["robot_signal"] = {"target": pending_proposal, "action_modifier": "update"}
                return output
            elif current_target:
                output["user_response"] = f"Agent: Confirmed. Keeping target on the {current_target}."
                return output

        # QuickPath’s minimalist full-word match interception
        # for example, if you enter the single word ‘cup’ directly
        if len(user_lower.split()) <= 2 and not any(kw in user_lower for kw in ["choose", "pick", "want", "not", "why", "which", "what", "change"]):
            matched = [obj for obj in objects if user_lower == obj.lower() or user_lower in obj.lower().split()]
            if len(matched) == 1:
                target_obj = matched[0]
                self.state.update_target(target_obj)
                output["user_response"] = f"Agent: Selected the {target_obj} via fast-track grounding."
                output["robot_signal"] = {"target": target_obj, "action_modifier": "update"}
                return output

        # Slow-track layer: Fully decentralised reasoning that integrates semantics and common sense in large language models
        self.state.add_history("User", user_raw)
        result = reason_about_input(user_raw, context)
        
        is_pure_query = result.get("is_pure_query", False)
        intent = result.get("intent", "unknown")
        object_name = result.get("object_name")
        explanation = result.get("explanation", "")

        # Align the nouns extracted from JSON slots by large language models with standard names of objects in the scene
        llm_preferred_target = None
        if object_name:
            obj_clean = object_name.lower().strip()
            for obj in objects:
                if obj_clean in obj.lower() or obj.lower() in obj_clean:
                    llm_preferred_target = obj
                    break

        # Active suggestion sentences for motion-capture robots:
        # If a reply contains key words such as ‘suggestion’, automatically flag the corresponding token.
        if any(w in explanation.lower() for w in ["suggest", "how about", "recommend"]):
            for obj in objects:
                if obj.lower() in explanation.lower():
                    self.state.set_proposal(obj)
                    break

        # As soon as the user utters keywords indicating a strong change of mind or a switch, the illusion of `pure_query` is physically severed.
        if any(keyword in user_lower for keyword in ["change back", "turn back", "the other", "dont want", "don't want", "not want", "change to", "not my favorite"]):
            is_pure_query = False
            intent = "select" if llm_preferred_target else "reject"

        # Whenever the conversation involves small talk, enquiries about one’s status, 
        # or questions about the reason, it is automatically converted to a pure enquiry status.
        is_action_command = intent in ["select", "reject", "delegate_choice"]
        if is_action_command and any(q_word in user_lower for q_word in ["why", "what we have", "what can", "which can"]):
            is_action_command = False
            is_pure_query = True

        if not is_action_command:
            llm_preferred_target = current_target   # to deprive them of the ability to modify the hardware’s focus

        # ---------- Pure queries / property enquiries / status enquiries path ----------
        if is_pure_query or not is_action_command:
            output["user_response"] = f"Agent: {explanation}" if explanation else f"Agent: Acknowledged your instruction."
            output["robot_signal"] = {"target": current_target, "action_modifier": "keep"}
            self.state.add_history("Agent", output["user_response"])
            return output

        # Highest priority: As long as the large model has derived a unique physical target string in `object_name`, this indicates that the resolution is complete.
        if llm_preferred_target and llm_preferred_target != current_target:
            self.state.update_target(llm_preferred_target)
            
            # To avoid the elementary mistake of saying ‘A’ and doing ‘A’, but ending up with duplicate concatenations
            if explanation and llm_preferred_target.lower() in explanation.lower() and "or" not in explanation:
                output["user_response"] = f"Agent: {explanation}"
            else:
                output["user_response"] = f"Agent: Targeting the {llm_preferred_target} as derived from your guidance."
                
            output["robot_signal"] = {"target": llm_preferred_target, "action_modifier": "update"}
            self.state.add_history("Agent", output["user_response"])
            return output

        # Downgrade Fuzzy Decision: Enable only when the large model is unable to provide a specific `object_name` slot
        if intent == "delegate_choice":
            text_targeted_obj = None
            if explanation:
                # If several physical objects appear in the text at the same time, the final conclusion should be based on the object that appears last.
                found_objs = [obj for obj in objects if obj.lower() in explanation.lower()]
                if len(found_objs) == 1:
                    text_targeted_obj = found_objs[0]
                elif len(found_objs) > 1:
                    text_targeted_obj = max(found_objs, key=lambda o: explanation.lower().rfind(o.lower()))
            
            final_chosen_obj = current_target
            if text_targeted_obj:
                final_chosen_obj = text_targeted_obj
            else:
                # If not specified, the first instance of the current object is selected by default.
                final_chosen_obj = objects[0]

            self.state.update_target(final_chosen_obj)
            
            # Announcement on Dynamic Text Circuit Breaker and Assembly
            if explanation and final_chosen_obj.lower() in explanation.lower() and "or" not in explanation:
                output["user_response"] = f"Agent: {explanation}"
            else:
                base_text = f"Agent: {explanation} " if explanation else "Agent: Acknowledged your delegation. "
                output["user_response"] = base_text + f"Therefore, I have autonomously decided to select the **{final_chosen_obj}** for you."
                
            output["robot_signal"] = {"target": final_chosen_obj, "action_modifier": "update"}
            self.state.add_history("Agent", output["user_response"])
            return output

        # Exclusive remedy for rescission. User: Give me the other one / I dont want it
        elif intent == "reject":
            if current_target is None: return output
            
            text_targeted_obj = None
            if explanation:
                found_objs = [obj for obj in objects if obj.lower() in explanation.lower() and obj != current_target]
                if found_objs:
                    text_targeted_obj = max(found_objs, key=lambda o: explanation.lower().rfind(o.lower()))

            if text_targeted_obj:
                new_target = text_targeted_obj
            else:
                alternatives = [obj for obj in objects if obj != current_target]
                new_target = alternatives[0] if alternatives else current_target

            self.state.update_target(new_target)
            output["user_response"] = f"Agent: Canceling action on {current_target}. Moving target to the alternative option: {new_target}."
            output["robot_signal"] = {"target": new_target, "action_modifier": "update"}
            self.state.add_history("Agent", output["user_response"])
            return output

        # It was an intention to take action, but it did not translate into a specific, actionable objective.
        # This usually occurs when the user mentions a name that does not appear in the context, 
        # or fails to specify which one they mean. There must never be a lack of output. 
        # The priority is to return the large model’s explanation to the user
        # if it does not provide one, dynamically generate a clarifying question based on the objects in the current context.
        if not output["user_response"]:
            if explanation:
                output["user_response"] = f"Agent: {explanation}"
            elif objects:
                obj_list = ", ".join(objects)
                output["user_response"] = (
                    f"Agent: I couldn't match that to anything on the table. "
                    f"Which one did you mean — {obj_list}?")
            else:
                output["user_response"] = "Agent: I don't see any objects on the table right now."
        # Maintain the current status quo; do not move the robotic arm; wait for the user to clarify.
        output["robot_signal"] = {"target": current_target, "action_modifier": "keep"}
        self.state.add_history("Agent", output["user_response"])
        return output