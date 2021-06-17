import math
import random

from func import calcDistMax, aleaCoord, Packet
from learn import Static
from Battery import Battery


class Node:
    """
    nodeid : id de la node
    period : temps moyen entre deux envoi de message (en ms)
    sf : spreading factor (entier compris entre 7 et 12)
    cr : coding rate (entier compris entre 1 et 4)
    bw : bandwith (125, 250 ou 500)
    coord : coordonées (x,y) de la node
    power : puissance d'émition des message en dB (entier compris entre 2 et 20 dB)
    packetLen : taille du packet
    freq : fréquence de la porteuse (860000000, 864000000, 868000000 hz)
            (peut valoir un entier entre 860000000 et 1020000000 par pas de 61hz, raport )
    packetSent : nombre de paquet envoyés
    packetLost : nombre de paquet ayant subi une colision
    messageLost : nombre de paquet définitivement perdu (7 collision pour un même paquet)
    validCombination : liste contenant les combinaison de paramètre valide
    """
    def __init__(self, nodeId: int, period: int, sensi, TX, packetLen=20, cr=1, bw=125, sf=7, power=14, coord=None, radius=200, algo=Static()):
        if coord is None:
            coord = aleaCoord(radius)
        self.nodeId = nodeId
        self.period = period
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.coord = coord
        self.power = power
        self.packetLen = packetLen
        self.freq = random.choice([860000000, 864000000, 868000000])
        self.packetSent = 0
        self.packetLost = 0
        self.messageLost = 0
        self.validCombination = self.checkCombination(sensi)
        self.waitTime = 0
        self.sendTime = 0
        self.TX = TX
        self.algo = algo
        self.active = False
        self.battery = Battery(1000)

    """
    construction de la liste contenant les combinaisons de paramètre valide (SF + Power)
    """
    def checkCombination(self, sensi) -> list:
        lTemp = []
        maxDist = calcDistMax(sensi)
        for i in range(len(maxDist)):
            for j in range(len(maxDist[i])):
                if maxDist[i][j] > math.sqrt((self.coord.x - 0) ** 2 + (self.coord.y - 0) ** 2):
                    lTemp.append([i + 7, j + 2])
        return lTemp

    def __str__(self):
        return "node : {:<4d} sf : {:<3d} packetSent : {:<6d} packetLost : {:<6d} messageLost : {:<6d} power : {:<3d}".format(
            self.nodeId, self.sf, self.packetSent, self.packetLost, self.messageLost, self.power)

    """
    Création d'un paquet corespondant aux paramètres de la node
    """
    def createPacket(self, idPacket) -> Packet:
        p = Packet(self.nodeId, self.packetLen, self.sf, self.cr, self.bw, self.coord, self.power, self.TX, idPacket)
        return p

    def setWaitTime(self, time: float, sendTime: float):
        self.waitTime = 99 * time
        self.sendTime = sendTime
