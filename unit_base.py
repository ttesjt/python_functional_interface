class FunctionalUnit:
    title = "Unnamed Unit"
    description = "No description provided."

    def setup(self, parent):
        raise NotImplementedError("Subclasses must implement .setup(parent)")