import random
from Packet import Packet, Point
from func import send, packetArrived
from graphic import nodePerSF, nbReemit, colectMeanPower
from simu import Event, Simu

#class de l'évent corespondant à l'envoie d'un packet
class sendPacketEvent(Event):
    # nodeId : id de la node qui envoie un message
    # idPacket : id du packet
    def __init__(self, nodeId: int, time: float, env: Simu, idPacket: int):
        super().__init__(time, env)
        self.nodeId = nodeId
        self.idPacket = idPacket

    def exec(self):
        node = self.env.envData["nodes"][self.nodeId]
        if (node.waitTime + node.sendTime) - self.time > 0:
            #print("erreur duty cycle ++")
            self.env.addEvent(sendPacketEvent(self.nodeId, node.waitTime + node.sendTime, self.env, self.idPacket))
            pass
        else:
            if node.active:
                #print("erreur la node est déja active")
                pass
            else:
                packet = node.createPacket(self.idPacket)
                send(packet, self.time, self.env)  # le packet est envoyé
                self.env.addEvent(receptPacketEvent(packet, self.time + packet.recTime, self.env))
                self.env.simTime = self.time
                self.env.envData["send"] += 1
                node.active = True

# class qui corespond au renvoie d'un packet ayant subie une collision
class ReSendPacketEvent(Event):
    # packet : packet ayant subbie une collision
    # idPacket : id du packet
    def __init__(self, time: float, env: Simu, packet: Packet, idPacket: int):
        super().__init__(time, env)
        self.packet = packet
        self.idPacket = idPacket

    def exec(self):
        node = self.env.envData["nodes"][self.packet.nodeId]
        if (node.waitTime + node.sendTime) - self.time > 0:
            #print("erreur duty cycle --")
            self.env.addEvent(ReSendPacketEvent(node.waitTime + node.sendTime, self.env, self.packet, self.idPacket))
        else:
            if node.active:
                #print("erreur la node est déja active")
                pass
            else:
                newPacket = self.env.envData["nodes"][self.packet.nodeId].createPacket(self.idPacket)
                newPacket.nbSend = self.packet.nbSend + 1
                send(newPacket, self.time, self.env)
                self.env.addEvent(receptPacketEvent(newPacket, self.time + newPacket.recTime, self.env))
                self.env.simTime = self.time
                self.env.envData["send"] += 1
                node.active = True

# class correspondant à la fin de la reception d'un packet
class receptPacketEvent(Event):
    # packet : packet qui à été recu
    def __init__(self, packet: Packet, time: float, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        packetArrived(self.packet, self.env)
        node = self.env.envData["nodes"][self.packet.nodeId]
        reemit = self.packet.nbSend
        nbReemit(self.env, self.packet)
        colectMeanPower(self.env, self.packet)
        if self.packet.lost:
            node.packetLost += 1
            reward = 0
            if self.packet.nbSend <= 7:
                reSendTime = node.waitTime + node.sendTime
                time = reSendTime + (reSendTime * random.uniform(0, 0.05))
                self.env.addEvent(ReSendPacketEvent(time, self.env, self.packet, self.packet.packetId))
            else :
                cost = self.packet.energyCost * self.packet.nbSend + 1
                reward = 1 - cost/5.9 # 1-(self.packet.energyCost/0.7)
                node.algo.update(node.algo.old_arm, reward)
                # sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend)
                arm = node.algo.select_arm(0.1)
                sf = node.validCombination[arm][0]
                power = node.validCombination[arm][1]
                nodePerSF(self.env, node.sf, sf)
                node.power = power
                node.sf = sf

            self.env.envData["collid"] += 1
        else :
            cost = self.packet.energyCost * self.packet.nbSend
            reward = 1 - cost/5.9 #1-(self.packet.energyCost/0.7)
            node.algo.update(node.algo.old_arm, reward)
            #sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend)
            arm = node.algo.select_arm(0.1)
            sf = node.validCombination[arm][0]
            power = node.validCombination[arm][1]
            nodePerSF(self.env, node.sf, sf)
            node.power = power
            node.sf = sf
        nodePerSF(self.env, node.sf, node.sf)



        if reemit == 0:
            time = self.time + random.expovariate(1.0 / node.period)
            self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env, self.packet.packetId + 1))
        self.env.simTime = self.time
        node.active = False

# Event qui permet de déplacer une node sur un point donné
class mooveEvent(Event):
    def __init__(self, time: int, env: Simu, p: Point, nodeId: int):
        super().__init__(time, env)
        self.p = p
        self.nodeId = nodeId

    def exec(self):
        self.env.envData["nodes"][self.nodeId].coord = self.p

# Event qui permet de placer une node à une distance donnée
class mooveDistEvent(Event):
    def __init__(self, time, env: Simu, dist, nodeId):
        super().__init__(time, env)
        self.dist = dist
        self.nodeId = nodeId

    def exec(self):
        nd = self.env.envData["nodes"][self.nodeId]
        sensi = self.env.envData["sensi"]
        nd.coord = Point(self.dist, 0)
        nd.validCombination = nd.checkCombination(sensi)
        nd.algo.__init__(n_arms=len(nd.validCombination))

# class qui corespond à l'évent gérant l'affichage du pourcentage d'execution
class timmerEvent(Event):
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
