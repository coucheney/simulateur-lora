from Packet import Packet

"""
objet corespondant a l'antenne 
"""
class BS:
    def __init__(self, idBS: int, coord):
        self.idBS = idBS
        self.coord = coord
        self.packetAtBS = []

    def __str__(self):
        return str(self.packetAtBS)

    # ajout d'un packet dans la liste de réception de l'antenne
    def addPacket(self, packet: Packet):
        self.packetAtBS.append(packet)

    # suppression d'un packet dans la liste de reception de l'antenne
    def removePacket(self, packet: Packet):
        self.packetAtBS.remove(packet)
