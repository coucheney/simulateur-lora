import random
from LoRa.Packet import Packet
from LoRa.func import send, packetArrived
from LoRa.graphic import packetPerSF, nbReemit
from simu import Event, Simu


class sendPacketEvent(Event):
    def __init__(self, nodeId, time, env: Simu):
        super().__init__(time, env)
        self.nodeId = nodeId

    def exec(self):
        packet = self.env.envData["nodes"][self.nodeId].createPacket()
        send(packet, self.time, self.env)  # le packet est envoyé
        self.env.addEvent(receptPacketEvent(packet, self.time + packet.recTime, self.env))
        self.env.simTime = self.time
        self.env.envData["send"] += 1

class ReSendPacketEvent(Event):
    def __init__(self, time, env: Simu, packet: Packet):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        newPacket = self.env.envData["nodes"][self.packet.nodeId].createPacket()
        newPacket.nbSend = self.packet.nbSend + 1
        send(newPacket, self.time, self.env)
        self.env.addEvent(receptPacketEvent(newPacket, self.time + newPacket.recTime, self.env))
        self.env.simTime = self.time
        self.env.envData["send"] += 1

cont = 0

class receptPacketEvent(Event):
    def __init__(self, packet: Packet, time, env: Simu):
        super().__init__(time, env)
        self.packet = packet

    def exec(self):
        node = self.env.envData["nodes"][self.packet.nodeId]
        lostPacket = False
        reemit = self.packet.nbSend
        if self.packet.lost:
            node.packetLost += 1
            lostPacket = True
            if self.packet.nbSend <= 7:
                rd = random.expovariate(1 / (self.packet.recTime * (self.packet.nbSend+1)))
                time = self.time + rd
                #self.env.addEvent(ReSendPacketEvent(time, self.env, self.packet))
            self.env.envData["collid"] += 1
        else:
            nbReemit(self.env, self.packet)

        packetArrived(self.packet, self.env)
        sf, power = node.algo.chooseParameter(self.packet.power, node.sf, lostPacket, node.validCombination, self.packet.nbSend)
        packetPerSF(self.env, node.sf, sf)
        node.power = power
        node.sf = sf

        if reemit == 0:
            time = self.time + random.expovariate(1.0 / node.period)
            self.env.addEvent(sendPacketEvent(self.packet.nodeId, time, self.env))
        self.env.simTime = self.time



class timmerEvent(Event):
    def __init__(self, time, env: Simu, maxTime):
        super().__init__(time, env)
        self.maxTime = maxTime

    def exec(self):
        print((self.time / self.maxTime)*100, "%")
        self.env.addEvent(timmerEvent(self.time + (self.maxTime/10), self.env, self.maxTime))



