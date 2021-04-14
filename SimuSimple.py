import math
import random
import simpy


class Packet:
    """
    nodeId : id de la node corespondant au packet
    packetLen : taille du packet
    sf : spreading factor
    cr : coding rate (def entre 1 et 4)
    bw : bandwidth
    recTime : temps de transmition du packet
    lost : booléen à True si le paquet est perdu
    nbSend : nombre de fois ou le même paquet à été retransmit
    """

    def __init__(self, nodeId, packetLen, sf, cr, bw):
        self.nodeId = nodeId
        self.packetLen = packetLen
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.recTime = self.airTime()  # temps de transmition
        self.lost = False
        self.nbSend = 0

    def __str__(self):
        return "nodeId : " + str(self.nodeId) + " sf : " + str(self.sf)

    """
    pl : nb d'octets du payload
    H : présence de l'entête (=0)
    DE : activation du low rate optimisation (=1)
    Npreamble : nb de symbole du préambule
    le calcul proviens de ce doc :  semtech-LoraDesignGuide_STD.pdf
    """
    def airTime(self):
        H = 0
        DE = 0
        nbreamble = 8

        Tsym = (2 ** self.sf) / self.bw
        payloadSymNb = 8 + max(
            math.ceil((8.0 * self.packetLen - 4.0 * self.sf + 28 + 16 - 20 * H) / (4.0 * (self.sf - 2 * DE))) * (
                    self.cr + 4), 0)  # nb de sybole dans le payload
        Tpreamble = nbreamble * Tsym
        Tpayload = payloadSymNb * Tsym
        Tpaquet = Tpreamble + Tpayload
        return Tpaquet


class Node:
    """
    nodeid : id de la node
    period : temps moyen entre deux envoi de message (en ms)
    packetLen : taille du packet
    packetSent : nombre de paquet envoyés
    packetLost : nombre de paquet ayant subi une colision
    messageLost : nombre de paquet définitivement perdu (7 collision pour un même paquet)
    """
    def __init__(self, nodeId, period, packetLen=20, cr=1, bw=125, sf=7):
        self.nodeId = nodeId
        self.period = period
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.packetLen = packetLen
        self.packetSent = 0
        self.packetLost = 0
        self.messageLost = 0

    def __str__(self):
        return "node : {:<4d} sf : {:<3d} packetSent : {:<6d} packetLost : {:<6d} messageLost : {:<6d}".format(
            self.nodeId, self.sf, self.packetSent, self.packetLost, self.messageLost)

    """
    Création d'un paquet corespondant aux paramètres de la node
    """
    def createPacket(self):
        return Packet(self.nodeId, self.packetLen, self.sf, self.cr, self.bw)


"""
Si les SF des deux paquets sont les mêmes, return True
"""
def sfCollision(p1, p2):
    if p1.sf == p2.sf:
        return True
    return False

"""
Gestion des collisions entre le paquet et les autre paquets qui sont en cours de reception
Collision gérée :
    - SF collision
"""
def collision(packet):
    if not packetsAtBS: # auccun paquet n'est en cours de réception
        return False
    else:
        for pack in packetsAtBS:
            if sfCollision(packet, pack):
                packet.lost = True
                pack.lost = True
                packetsAtBS.remove(pack)
                return True
        return False


def send(packet):
    nodes[packet.nodeId].packetSent += 1
    if not collision(packet):
        packetsAtBS.append(packet)


def packetArrived(packet):
    if packet in packetsAtBS:
        packetsAtBS.remove(packet)

"""
prosessus de transmition d'un paquet
"""
def transmit(packet, period):
    time = random.expovariate(1.0 / float(period))
    yield env.timeout(time)  # date de début de l'envoie du packet
    send(packet)  # le packet est envoyé

    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        global stop
        stop = packet.nodeId
        newPacket = nodes[packet.nodeId].createPacket()
        newPacket.nbSend += 1
        #env.process(reTransmit(newPacket))
    else:
        packetArrived(packet)
    env.process(transmit(nodes[packet.nodeId].createPacket(), period))

"""
processus qui permet le retransmition d'un paquet, si celui ci a subi une colision
"""
def reTransmit(packet):
    time = 0
    yield env.timeout(time)  # date de début de l'envoie du packet
    send(packet)  # le packet est envoyé

    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        if packet.nbSend < 7:
            global stop
            stop = packet.nodeId
            newPacket = nodes[packet.nodeId].createPacket()
            newPacket.nbSend = packet.nbSend + 1
            env.process(reTransmit(newPacket))
        else:
            nodes[packet.nodeId].messageLost += 1
    else:
        packetArrived(packet)

"""
début de la simulation 
"""
def startSimulation(simTime, nbStation, period, packetLen):
    # créatoin des nodes, des paquets et des évent d'envoi des messages
    for idStation in range(nbStation):
        node = Node(idStation, period, packetLen)
        nodes.append(node)
        env.process(transmit(node.createPacket(), node.period))
    #lancement de la simulation
    while env.peek() < simTime:
        global stop
        if not stop == -1: # pause dans la simulation : se fait à chaque faois q'un packet est perdu
            if aleaAdapt:
                nodes[stop].sf = random.randint(7, 12) # le sf est redéfinit aléatoirement
            stop = -1
        env.step()


def main():
    simTime = 1800000000  # Temps de simulation en ms 00
    nbStation = 100      # Nombre de node dans le réseau
    period = 1800000      # Interval de temps moyen entre deux message
    packetLen = 20

    # lancement de la simulation
    startSimulation(simTime, nbStation, period, packetLen)

    # affichage des résultats
    sumSend = 0
    sumLost = 0
    sumMessageLost = 0
    for node in nodes:
        sumSend += node.packetSent
        sumLost += node.packetLost
        sumMessageLost += node.messageLost
    print()
    for node in nodes:
        print(str(node))
    print()
    print("sumPacketSend :", sumSend)
    print("sumPacketLost :", sumLost)
    print("sumMessageLost :", sumMessageLost)

"""
Variables globales pour la simulation
"""
aleaAdapt = True
env = simpy.Environment()
packetsAtBS = []
nodes = []
station = []
stop = -1


main()
