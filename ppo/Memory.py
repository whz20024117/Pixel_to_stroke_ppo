class Memory:
    def __init__(self):
        self.action = []
        self.state = []
        self.state_ = []
        self.reward = []
        self.terminal = []

    def reset(self):
        self.action.clear()
        self.state.clear()
        self.state_.clear()
        self.reward.clear()
        self.terminal.clear()

    def count(self):
        return len(self.action)

    def add(self, s, a, r, s_, t):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.state_.append(s_)
        self.terminal.append(t)




