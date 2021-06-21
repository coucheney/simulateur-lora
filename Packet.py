import math
import random

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Packet:
    """
    nodeId : id de la node corespondant au packet
    packetLen : taille du packet
    sf : spreading factor (entier compris entre 7 et 12)
    cr : coding rate (entier compris entre 1 et 4)
    bw : bandwith (125, 250 ou 500)
    recTime : temps de transmition du packet
    lost : booléen à True si le paquet est perdu
    nbSend : nombre de fois ou le même paquet à été retransmit
    rssi : puissance du packet au niveau de l'antenne
    sendDate : date à laquelle le paquet commence à être envoyé, prend une valeurs lors de l'envoi du paquet
    """


    def __init__(self, nodeId: int, packetLen: int, sf: int, cr: int, bw: int, pNode: Point, power: int, TX, packetId):
        self.nodeId = nodeId
        self.packetId = packetId
        self.packetLen = packetLen
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.recTime = self.airTime()  # temps de transmition
        self.lost = False
        self.nbSend = 0
        self.coordNode = pNode
        self.rssi = calcRssi(pNode, Point(0, 0), power)  # [0, 0] pour le moment l'antenne est placée en 0,0
        self.sendDate = None
        self.freq = random.choice([860000000, 864000000, 868000000])
        self.power = power
        self.energyCost = (self.recTime/3600000) * (TX[int(self.power) + 2])

    def __str__(self):
        return "nodeId : " + str(self.nodeId) + " sf : " + str(self.sf) + " rssi : " + str(self.rssi)

    """
        calcul du temps pendant lequel le paquet serra transmit
        H : présence de l'entête (=0)
        DE : activation du low rate optimisation (=1)
        Npreamble : nb de symbole du préambule
        le calcul proviens de ce doc :  semtech-LoraDesignGuide_STD.pdf
        """

    def airTime(self) -> float:
        H = 0  #
        DE = 0
        nbreamble = 8
        Tsym = (2 ** self.sf) / self.bw
        payloadSymNb = 8 + max(
            math.ceil((8.0 * self.packetLen - 4.0 * self.sf + 28 + 16 - 20 * H) / (4.0 * (self.sf - 2 * DE))) * (
                    self.cr + 4), 0)  # nb de sybole dans le payload
        Tpreamble = (nbreamble + 4.25) * Tsym  # temps d'envoi du préambule
        Tpayload = payloadSymNb * Tsym  # temps d'envoi du reste du packet
        Tpaquet = Tpreamble + Tpayload
        # print(self.packetLen, self.sf, Tsym)
        # print(Tpaquet, Tpreamble, Tpayload)
        # print(Tpaquet, self.sf)
        return Tpaquet

""" 
fonction pour calculer la perte de puissance du signal en fonction de la distance 
Renvoi la puissance du signal lorsqu'il est capté par l'antenne
Utilisation du log-distance path loss model
Les variable gamma,lpld0 sont obtenues de manière empirique
Le modèle de loraSim simule la perte de puissance dans une zone dense
"""
def calcRssi(pNode: Point, pBase: Point, power: int) -> float:
    gamma = 2.08
    lpld0 = 127.41
    d0 = 40
    GL = 0  # vaut toujours 0 (dans LoraSim et LoraFREE)
    d = math.sqrt((pNode.x - pBase.x) ** 2 + (pNode.y - pBase.y) ** 2)  # distance entre l'antenne et la node
    lpl = lpld0 + 10 * gamma * math.log10(d / d0)  # possible d'y ajouter une variance (par ex +/- 2 dans LoraFree)
    prx = power - GL - lpl
    return prx
