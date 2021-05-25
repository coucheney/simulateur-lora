import random

from LoRa.func import send, packetArrived
from LoRa.graphic import packetPerSF
from simu import Event, Simu


class sendPacketEvent(Event):
    def __init__(self, nodeId, time, env: Simu):
        super().__init__(time, env)
        self.nodeId = nodeId
        self.packet = env.envData["nodes"][nodeId].createPacket()

    def exec(self):
        send(self.packet, self.time, self.env)  # le packet est envoyé
        self.env.addEvent(receptPacketEvent(self.packet, self.time + self.packet.recTime, self.env))
        self.env.simTime = self.time
        packetPerSF(self.env, self.packet)


class receptPacketEvent(Event):
    def __init__(self, packet, time, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        if self.packet.lost:
            self.env.envData["nodes"][self.packet.nodeId].packetLost += 1
            stop = self.packet.nodeId
            colid = True

            newPacket = self.env.envData["nodes"][self.packet.nodeId].createPacket()
            newPacket.nbSend += 1
            #env.process(reTransmit(newPacket))
        else:
            stop = self.packet.nodeId
            colid = False
        messStop = self.packet
        packetArrived(self.packet, self.env)
        time = self.time + random.expovariate(1.0 / self.env.envData["nodes"][self.packet.nodeId].period)
        self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env))
        self.env.simTime = self.time

class timmerEvent(Event):
    def __init__(self, time, env: Simu, maxTime):
        super().__init__(time, env)
        self.maxTime = maxTime

    def exec(self):
        print((self.time / self.maxTime)*100, "%")
        self.env.addEvent(timmerEvent(self.time + (self.maxTime/10), self.env, self.maxTime))



