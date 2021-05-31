

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
        cont = len(self.events) - 1
        if self.events:
            while (not cont == -1) and int(self.events[cont].time) < int(event.time):
                cont -= 1
        self.events.insert(cont+1, event)

    def nextEvent(self):
        if self.events:
            nextEvent = self.events.pop()
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
        print(self.time)
