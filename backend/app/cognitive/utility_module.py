class UtilityModule:
    def compute_utility(self, valence: float):
        base = 1.0

        if valence < 0:
            base += 0.5

        return base
