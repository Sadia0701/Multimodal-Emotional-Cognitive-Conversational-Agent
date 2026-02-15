from typing import List


class GoalModule:
    def __init__(self):
        self.active_goals: List[str] = []

    def add_goal(self, goal: str):
        if goal not in self.active_goals:
            self.active_goals.append(goal)

    def remove_goal(self, goal: str):
        if goal in self.active_goals:
            self.active_goals.remove(goal)

    def get_primary_goal(self):
        if self.active_goals:
            return self.active_goals[0]
        return None
