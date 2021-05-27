import random

from LoRa.func import send, packetArrived
from simu import Event, Simu


class sendPacketEvent(Event):
    def __init__(self, nodeId, time, env: Simu):
        super().__init__(time, env)
        self.nodeId = nodeId
        self.packet = env.envData["nodes"][nodeId].createPacket()

    def exec(self):
        send(self.packet, self.time, self.env)  # le packet est envoyé
        self.env.envData["send"] += 1
        self.env.addEvent(receptPacketEvent(self.packet, self.time + self.packet.recTime, self.env))
        self.env.simTime = self.time


class receptPacketEvent(Event):
    def __init__(self, packet, time, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        node = self.env.envData["nodes"][self.packet.nodeId]
        lostPacket = False
        if self.packet.lost:
            node.packetLost += 1
            lostPacket = True
            self.env.envData["collid"] += 1
            newPacket = node.createPacket()
            newPacket.nbSend += 1
            #env.process(reTransmit(newPacket))
        packetArrived(self.packet, self.env)
        time = self.time + random.expovariate(1.0 / node.period)
        self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env))
        self.env.simTime = self.time

        sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend)
        node.power = power
        node.sf = sf

class timmerEvent(Event):
    def __init__(self, time, env: Simu, maxTime):
        super().__init__(time, env)
        self.maxTime = maxTime

    def exec(self):
        print((self.time / self.maxTime)*100, "%")
        self.env.addEvent(timmerEvent(self.time + (self.maxTime/10), self.env, self.maxTime))



