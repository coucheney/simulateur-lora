from Packet import Packet


class BS:
    def __init__(self, idBS, coord):
        self.idBS = idBS
        self.coord = coord
        self.packetAtBS = []

    def __str__(self):
        return str(self.packetAtBS)

    def addPacket(self, packet: Packet):
        self.packetAtBS.append(packet)

    def removePacket(self, packet):
        self.packetAtBS.remove(packet)
