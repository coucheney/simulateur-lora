

class Simu:
    def __init__(self):
        self.events = []
        self.simTime = 0
        self.envData = {}

    def __str__(self):
        ret = ""
        for ev in self.events:
            ret = ret + " " + str(ev)
        return ret

    def addEvent(self, event):
        self.events.append(event)
        self.events.sort(reverse=True)

    def nextEvent(self):
        if self.events:
            nextEvent = self.events.pop(-1)
            nextEvent.exec()

    def addData(self, value, key):
        self.envData[key] = value

class Event:
    def __init__(self, time, env: Simu):
        self.time = time
        self.env = env

    def __lt__(self, other):
        return self.time < other.time

    def __str__(self):
        return "time : " + str(self.time)

    def exec(self):
        pass
