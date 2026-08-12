# dialogue/state_manager.py

class StateManager:
    """
    The simulation side of the dialogue state + scene manifest
    is populated in real time at each turn (names / colours / shapes of objects on the table / table partitions / relative positions), and injected via `set_objects_manifest()`.
    `get_context()` continues to automatically concatenate all attributes of any dimension into a single text string to feed to the large model.
    """

    def __init__(self, current_task: str = "Pick an object"):
        self.current_task = current_task
        self.objects_manifest = {}

        self.current_target = None
        self.dialogue_history = []
        self.pending_proposal = None

    # Scene injection, called on each round on the simulation side
    def set_objects_manifest(self, manifest: dict):
        # Replace the current manifest with the manifest obtained from a simulated real-time scan
        # clear the manifest if `manifest` is `None`.
        self.objects_manifest = manifest or {}

    def set_task(self, task: str):
        if task:
            self.current_task = task

    def reset_session(self):
        # Reset the scene: Clear the target, suggestions and conversation history, and start a new conversation
        self.current_target = None
        self.pending_proposal = None
        self.dialogue_history = []

    # Context is packaged and fed to the inference layer
    def get_context(self):
        object_names = list(self.objects_manifest.keys())

        manifest_text = ""
        for name, attrs in self.objects_manifest.items():
            attr_strings = [f"{k}: {v}" for k, v in attrs.items()]
            manifest_text += f"- {name}: [{', '.join(attr_strings)}]\n"
        if not manifest_text:
            manifest_text = "- (no objects detected yet)\n"

        return {
            "task": self.current_task,
            "objects": object_names,
            "objects_manifest_text": manifest_text,
            "current_target": self.current_target,
            "history": self.dialogue_history,
            "pending_proposal": self.pending_proposal,
        }


    def update_target(self, target):
        self.current_target = target
        self.pending_proposal = None

    def set_proposal(self, proposal_obj):
        self.pending_proposal = proposal_obj

    def add_history(self, speaker, text):
        self.dialogue_history.append({"speaker": speaker, "text": text})
        self.dialogue_history = self.dialogue_history[-10:]