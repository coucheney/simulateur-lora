
# classe correspondant au simulateur et a son environement
class Simu:
    # L'attribut events est la liste des évènement en attente dans l'échéancier
    # L'atribut Simtime contient le temp ou en est la simulation
    # L'atribut envData contient un dictionnaire ou sont stockés les différentes données de la simulation
    def __init__(self):
        self.events = []
        self.simTime = 0
        self.envData = {}

    def __str__(self):
        ret = ""
        for ev in self.events:
            ret = ret + " " + str(ev)
        return ret

    #ajout d'un event dans l'échéancier
    # envent : event a ajouter dans l'échéancier
    def addEvent(self, event):
        if self.events:
            cont = len(self.events)
            while cont > 0 and self.events[cont-1].time < event.time:
                cont -= 1
            self.events.insert(cont, event)
        else:
            self.events.append(event)

    # méthode qui renvoie l'évènement suivant
    def nextEvent(self):
        if self.events:
            nextEvent = self.events.pop()
            if self.events:
                if nextEvent.time > self.events[-1].time:
                    print(nextEvent.time, "---", self.events[-1])
            nextEvent.exec()

    #ajoute une entrée dans le dictionnaire contenant les données de la simulation
    # value : variable ajoutée dans le dictionnaire
    # key : clée utilisée dans le dictionnaire
    def addData(self, value, key):
        self.envData[key] = value

# super-class corespondant aux evenements
class Event:
    def __init__(self, time, env: Simu):
        self.time = time
        self.env = env

    # redéfinition du  symbole <, permet de comparer la date des objet Event (ou héritant d'Event)
    def __lt__(self, other) -> bool:
        return self.time < other.time

    def __str__(self):
        return "time : " + str(self.time)

    # fonction qui permet d'executer les action de l'évènement
    def exec(self):
        pass
