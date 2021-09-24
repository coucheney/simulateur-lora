import random

from Packet import Packet, Point
from func import send, packetArrived
from graphic import nodePerSF, nbReemit, colectMeanPower, colectData, getlog
from simu import Event, Simu

#class de l'évent corespondant à l'envoie d'un packet
class sendPacketEvent(Event):
    # nodeId : id de la node qui envoie un message
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    # idPacket : id du packet
    def __init__(self, nodeId: int, time: float, env: Simu, idPacket: int):
        super().__init__(time, env)
        self.nodeId = nodeId
        self.idPacket = idPacket

    def exec(self):
        node = self.env.envData["nodes"][self.nodeId]
        # décalage de l'event si le duty cycle n'est pas respecté
        if (node.waitTime + node.sendTime) - self.time > 0:
            self.env.addEvent(sendPacketEvent(self.nodeId, node.waitTime + node.sendTime, self.env, self.idPacket))
        else:
            # le noeud est déja en cours d'utilisation
            if node.active:
                # le paquet est perdu et le suivant et l'event du packet suivnat est crée
                self.env.envData["firstSend"][self.nodeId] += 1
                time = self.time + random.expovariate(1.0 / node.period)
                self.env.addEvent(sendPacketEvent(self.nodeId, time, self.env, self.idPacket + 1))
                #self.env.envData["nodes"][self.nodeId].waitPacket.append(node.createPacket(self.idPacket))
            else:
                # envoie du packet
                packet = node.createPacket(self.idPacket)
                send(packet, self.time, self.env)
                self.env.addEvent(receptPacketEvent(packet, self.time + packet.recTime, self.env))
                self.env.simTime = self.time
                self.env.envData["send"] += 1
                node.active = True

# class qui corespond au renvoie d'un packet ayant subie une collision
class ReSendPacketEvent(Event):
    # nodeId : id de la node qui envoie un message
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    # idPacket : id du packet
    def __init__(self, time: float, env: Simu, packet: Packet, idPacket: int):
        super().__init__(time, env)
        self.packet = packet
        self.idPacket = idPacket

    def exec(self):
        node = self.env.envData["nodes"][self.packet.nodeId]
        # décalage de l'event si le duty cycle n'est pas respecté
        if (node.waitTime + node.sendTime) - self.time > 0:
            self.env.addEvent(ReSendPacketEvent(node.waitTime + node.sendTime, self.env, self.packet, self.idPacket))
        else:
            # le noeud est déja en cours d'utilisation
            if node.active:
                pass
                #self.env.envData["nodes"][self.packet.nodeId].waitPacket.append(self.packet)
            else:
                # renvoie du packet
                newPacket = self.env.envData["nodes"][self.packet.nodeId].createPacket(self.idPacket)
                newPacket.nbSend = self.packet.nbSend + 1
                send(newPacket, self.time, self.env)
                self.env.addEvent(receptPacketEvent(newPacket, self.time + newPacket.recTime, self.env))
                self.env.simTime = self.time
                self.env.envData["send"] += 1
                node.active = True

# class correspondant à la fin de la reception d'un packet
class receptPacketEvent(Event):
    # packet : paquet étant reçu
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    def __init__(self, packet: Packet, time: float, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        packetArrived(self.packet, self.env)
        node = self.env.envData["nodes"][self.packet.nodeId]
        lostPacket = False
        reemit = self.packet.nbSend
        nbReemit(self.env, self.packet)
        colectMeanPower(self.env, self.packet)
        colectData(self.env, self.packet, self.time)

        if self.packet.lost:        # le paquet à été perdu
            node.packetLost += 1
            lostPacket = True
            if self.packet.nbSend <= 7:     # Le nombre max de réémition n'a pas été atteint
                reSendTime = node.waitTime + node.sendTime
                time = reSendTime + (reSendTime * random.uniform(0, 0.05))
                self.env.addEvent(ReSendPacketEvent(time, self.env, self.packet, self.packet.packetId))

        #Partie DQN: le booléen est par défaut en False. Pour faire fonctionner le DQN, il faut l'activer dans la fonction initSimulation (main.py)
        if self.env.envData["activateBaseLearning"]:
            if not self.packet.lost:  # Le paquet est reçu par la station
                if self.env.envData["BS"].first[node.nodeId] == True :
                    sensi = self.env.envData["sensi"][self.packet.sf - 7, [125, 250, 500].index(125) + 1]
                    change = (True if self.packet.nbSend == 7 else False)
                    nodesf, nodepower = self.env.envData["BS"].recommandation2(self.packet.rssi, sensi, self.packet.sf,
                                                                               self.packet.nodeId,
                                                                               change)
                    node.set_parameter4(nodesf, nodepower)

        # changement des paramètres sf et power
        sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend, energyCost=self.packet.energyCost)
        nodePerSF(self.env, node.sf, sf, node.power, power)
        node.power = power
        node.sf = sf
        getlog(self.env, self.packet.nodeId, self.packet)

        # création de l'évent pour l'envoie du paquets suivants
        if reemit == 0:
            time = self.time + random.expovariate(1.0 / node.period)
            self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env, self.packet.packetId + 1))
            self.env.envData["firstSend"][self.packet.nodeId] += 1
        self.env.simTime = self.time

        node.active = False

# Event qui permet de déplacer une node sur un point donné
class mooveEvent(Event):
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    # nodeId : id du noeud
    # p : point sur lequel on déplace le noeud
    def __init__(self, time: int, env: Simu, p: Point, nodeId: int):
        super().__init__(time, env)
        self.p = p
        self.nodeId = nodeId

    def exec(self):
        if self.nodeId < len(self.env.envData["nodes"]):
            self.env.envData["nodes"][self.nodeId].coord = self.p
        else:
            print("erreur, la Node" + str(self.nodeId) + " n'existe pas")

# Event qui permet de placer une node à une distance donnée
class mooveDistEvent(Event):
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    # nodeId : id du noeud
    # dist : distance à laquelle on déplace le noeud
    def __init__(self, time, env: Simu, dist: int, nodeId: int):
        super().__init__(time, env)
        self.dist = dist
        self.nodeId = nodeId

    def exec(self):
        if self.nodeId < len(self.env.envData["nodes"]):
            nd = self.env.envData["nodes"][self.nodeId]
            sensi = self.env.envData["sensi"]
            nd.coord = Point(self.dist, 0)
            nd.validCombination = nd.checkCombination(sensi)
        else:
            print("erreur, la Node" + str(self.nodeId) + " n'existe pas")

class addNodeEvent(Event):
    def __init__(self, time, env: Simu, nodes):
        super().__init__(time, env)
        self.newNodes = nodes

# class qui corespond à l'évent gérant l'affichage du pourcentage d'execution
class timmerEvent(Event):
    # time : date ou l'envent va s'éxecuter
    # env : environement de la simulation
    # maxTime : temps de la simulation
    # count : conteur du nombre de fois ou cet event à été appelé
    def __init__(self, time: float, env: Simu, maxTime: int, count: int):
        super().__init__(time, env)
        self.maxTime = maxTime
        self.count = count

    def exec(self):
        self.env.addEvent(timmerEvent(self.time + (self.maxTime/10), self.env, self.maxTime, self.count + 1))
        self.env.simTime = self.time
        nbChar = 10
        loadBar = "["
        for i in range(self.count):
            loadBar += "#"
        for i in range(nbChar - self.count):
            loadBar += "."
        loadBar += "]"
        print(loadBar, self.count*10, "%")
