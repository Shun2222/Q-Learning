
class Prey:
    def __init__(self, id, actionNum):
        self.id = id
        self.isCaptured = False
        self.actionNum = actionNum

    def action(self):
        return np.random.randint(actionNum)

    def reset(self):
        self.isCaptured = False:
