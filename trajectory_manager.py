from collections import deque
class TrajectoryManager:

    def __init__(self):

        self.history = {
            1:deque(maxlen=5000),
            2:deque(maxlen=5000)
        }

    def update(
        self,
        player_number,
        court_x,
        court_y
    ):

        self.history[player_number].append(
            (court_x, court_y)
        )

    def get_history(
        self,
        player_number
    ):

        return list(
            self.history[player_number]
        )
