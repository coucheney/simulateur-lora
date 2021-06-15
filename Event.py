import random
from Packet import Packet, Point
from func import send, packetArrived
from graphic import nodePerSF, nbReemit, colectMeanPower
from simu import Event, Simu


class sendPacketEvent(Event):
    def __init__(self, nodeId, time, env: Simu, idPacket):
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

class ReSendPacketEvent(Event):
    def __init__(self, time, env: Simu, packet: Packet, idPacket):
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

class receptPacketEvent(Event):
    def __init__(self, packet: Packet, time, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        packetArrived(self.packet, self.env)
        node = self.env.envData["nodes"][self.packet.nodeId]
        lostPacket = False
        reemit = self.packet.nbSend
        nbReemit(self.env, self.packet)
        colectMeanPower(self.env, self.packet)
        if self.packet.lost:
            node.packetLost += 1
            lostPacket = True
            if self.packet.nbSend <= 7:
                reSendTime = node.waitTime + node.sendTime
                time = reSendTime + (reSendTime * random.uniform(0, 0.05))
                self.env.addEvent(ReSendPacketEvent(time, self.env, self.packet, self.packet.packetId))
            self.env.envData["collid"] += 1

        sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend)
        nodePerSF(self.env, node.sf, sf)
        node.power = power
        node.sf = sf

        if reemit == 0:
            time = self.time + random.expovariate(1.0 / node.period)
            self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env, self.packet.packetId + 1))
        self.env.simTime = self.time
        node.active = False



class timmerEvent(Event):
    def __init__(self, time, env: Simu, maxTime):
        super().__init__(time, env)
        self.maxTime = maxTime

    def exec(self):
        print((self.time / self.maxTime)*100, "%")
        self.env.addEvent(timmerEvent(self.time + (self.maxTime/10), self.env, self.maxTime))
        self.env.simTime = self.time

class mooveEvent(Event):
    def __init__(self, time, env: Simu, p: Point, nodeId):
        super().__init__(time, env)
        self.p = p
        self.nodeId = nodeId

    def exec(self):
        self.env.envData["nodes"][self.nodeId].coord = self.p

class mooveDistEvent(Event):
    def __init__(self, time, env: Simu, dist, nodeId):
        super().__init__(time, env)
        self.dist = dist
        self.nodeId = nodeId

    def exec(self):
        self.env.envData["nodes"][self.nodeId].coord = Point(self.dist, 0)


