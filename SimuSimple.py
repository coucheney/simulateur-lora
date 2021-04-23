import math
import random
import simpy
import numpy as np
import matplotlib.pyplot as plt


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
    def __init__(self, nodeId, packetLen, sf, cr, bw, pNode, power):
        self.nodeId = nodeId
        self.packetLen = packetLen
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.recTime = self.airTime()  # temps de transmition
        self.lost = False
        self.nbSend = 0
        self.rssi = calcRssi(pNode, [0, 0], power)
        self.sendDate = None

    def __str__(self):
        return "nodeId : " + str(self.nodeId) + " sf : " + str(self.sf) + " rssi : " + str(self.rssi)

    """
    calcul du temps pendant lequel le paquet serra transmit
    H : présence de l'entête (=0)
    DE : activation du low rate optimisation (=1)
    Npreamble : nb de symbole du préambule
    le calcul proviens de ce doc :  semtech-LoraDesignGuide_STD.pdf
    """
    def airTime(self):
        H = 0  #
        DE = 0
        nbreamble = 8
        Tsym = (2 ** self.sf) / self.bw
        payloadSymNb = 8 + max(math.ceil((8.0 * self.packetLen - 4.0 * self.sf + 28 + 16 - 20 * H) / (4.0 * (self.sf - 2 * DE))) * (self.cr + 4), 0)  # nb de sybole dans le payload
        Tpreamble = (nbreamble + 4.25) * Tsym   # temps d'envoi du préambule
        Tpayload = payloadSymNb * Tsym          # temps d'envoi du reste du packet
        Tpaquet = Tpreamble + Tpayload
        return Tpaquet


""" 
fonction pour calculer la perte de puissance du signal en fonction de la distance 
Renvoi la puissance du signal lorsqu'il est capté par l'antenne
Utilisation du log-distance path loss model
Les variable gamma,lpld0 sont obtenues de manière empirique
Le modèle de loraSim simule la perte de puissance dans une zone dense
"""
def calcRssi(pNode, pBase, power):
    gamma = 2.08
    lpld0 = 127.41
    d0 = 40
    GL = 0  # vaut toujours 0 (dans LoraSim et LoraFREE)
    d = math.sqrt((pNode[0] - pBase[0]) ** 2 + (pNode[1] - pBase[1]) ** 2)  # distance entre l'antenne et la node
    lpl = lpld0 + 10 * gamma * math.log10(d / d0)  # possible d'y ajouter une variance (par ex +/- 2 dans LoraFree)
    prx = power - GL - lpl
    return prx


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
            (peut valoir un entier entre 860000000 et 1020000000 par pas de 61hz, raport stage)
    packetSent : nombre de paquet envoyés
    packetLost : nombre de paquet ayant subi une colision
    messageLost : nombre de paquet définitivement perdu (7 collision pour un même paquet)
    """
    def __init__(self, nodeId, period, packetLen=20, cr=4, bw=125, sf=12, power=14):
        self.nodeId = nodeId
        self.period = period
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.coord = aleaCoord()
        self.power = power
        self.packetLen = packetLen
        self.freq = random.choice([860000000, 864000000, 868000000])
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
        p = Packet(self.nodeId, self.packetLen, self.sf, self.cr, self.bw, self.coord, self.power)
        return p

"""
fonction qui renvoie des coordonée aléatoire dans un cercle 
ici 200 mètre 
"""
def aleaCoord():
    a = random.random()
    b = random.random()
    radius = 200
    if b < a:
        a, b = b, a
    posx = b * radius * math.cos(2 * math.pi * a / b)
    posy = b * radius * math.sin(2 * math.pi * a / b)
    return [posx, posy]


"""
Si les SF des deux paquets sont les mêmes, return True
"""
def sfCollision(p1, p2):
    if p1.sf == p2.sf:
        return True
    return False


######################################################################## Pas utilisé questionnement en cour
# frequencyCollision, conditions
#
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
#   fonction utilisée entre deux paquet sur le même SF
#   avec les paramètres 868.10, 868.30 et 868.50, il ne peut pas y avoir de collision si p1.freq != p2.freq
def frequencyCollision(p1, p2):
    if abs(p1.freq - p2.freq) <= 120 and (p1.bw == 500 or p2.bw == 500):
        return True
    elif abs(p1.freq - p2.freq) <= 60 and (p1.bw == 250 or p2.bw == 250):
        return True
    elif abs(p1.freq - p2.freq) <= 30:
        return True
    return False
########################################################################

"""
colision de fréquence entre les paquets 
la différence de fréquence minimal entre deux paquets est dans loraSim est de 6 dB
Le paquet 1 est celui qui arrive
"""
def powerCollision(p1, p2):
    powerThreshold = 6  # dB
    if abs(p1.rssi - p2.rssi) < powerThreshold:  # La puissance des deux message est trop proche, les deux paquet sont perdu
        return [p1, p2]
    elif p1.rssi < p2.rssi:  # le paquet qui arrive est moins puissant que le paquet déja présent, le paquet qui arrive est perdu
        return [p1]
    else:               # le paquet qui vient d'arriver est plus puissant
        return [p2]

"""
collision au niveau des timming des paquets
retourne False si au moins 5 symbole du preambule du paquet 1, sont emis après la fin du paquet 2
True sinon (collision entre les deux paquets) 
"""
def timingCollision(p1, p2):
    # assuming 8 preamble symbols
    Npream = 8
    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2 ** p1.sf / p1.bw * (Npream - 5)
    p2_end = p2.sendDate + p2.recTime
    p1_cs = env.now + Tpreamb
    if p1_cs < p2_end:      # il faut que l'antenne puisse capter au moins les 5 derniers symbole du préambule
        return True
    return False


"""
Gestion des collisions entre le paquet et les autre paquets qui sont en cours de reception
le packet est marqué perdu si sa puissance est inférieur à la sensibilité 
Collision gérée :
    + SF collision
    + timmingCollision
    + powerColision
"""
def collision(packet):
    sensitivity = sensi[packet.sf - 7, [125, 250, 500].index(packet.bw) + 1]
    if packet.rssi < sensitivity:  # La puissance du paquet est plus faible que la sensivitivity
        packet.lost = True
        return True

    if packetsAtBS:  # au moins un paquet est en cours de réception
        for pack in packetsAtBS:
            if sfCollision(packet, pack) and timingCollision(packet, pack):
                packetColid = powerCollision(packet, pack)
                for p in packetColid:
                    p.lost = True
                packetsAtBS.remove(pack)
                return True
    return False


def send(packet, sendDate):
    nodes[packet.nodeId].packetSent += 1
    packet.sendDate = sendDate
    if not collision(packet):
        packetsAtBS.append(packet)


def packetArrived(packet):
    if packet in packetsAtBS:
        global messStop
        messStop = packet
        packetsAtBS.remove(packet)


"""
prosessus de transmition d'un paquet
"""
def transmit(packet, period):
    time = random.expovariate(1.0 / float(period))
    yield env.timeout(time)  # date de début de l'envoie du packet
    send(packet, env.now)  # le packet est envoyé
    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    global stop
    global colid
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        stop = packet.nodeId
        colid = True
        newPacket = nodes[packet.nodeId].createPacket()
        newPacket.nbSend += 1
        env.process(reTransmit(newPacket))
    else:
        packetArrived(packet)
        stop = packet.nodeId
        colid = False
    env.process(transmit(nodes[packet.nodeId].createPacket(), period))


"""
processus qui permet le retransmition d'un paquet, si celui ci a subi une colision
"""
def reTransmit(packet):
    time = 0
    yield env.timeout(time)  # date de début de l'envoie du packet
    send(packet, env.now)  # le packet est envoyé

    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    global stop
    global colid
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        if packet.nbSend < 7:
            stop = packet.nodeId
            colid = True
            newPacket = nodes[packet.nodeId].createPacket()
            newPacket.nbSend = packet.nbSend + 1
            env.process(reTransmit(newPacket))
        else:
            nodes[packet.nodeId].messageLost += 1
    else:
        packetArrived(packet)
        stop = packet.nodeId
        colid = False


"""
début de la simulation 
"""
def startSimulation(simTime, nbStation, period, packetLen):
    # créatoin des nodes, des paquets et des évent d'envoi des messages
    for idStation in range(nbStation):
        node = Node(idStation, period, packetLen, sf=7)
        nodes.append(node)
        env.process(transmit(node.createPacket(), node.period))
    # lancement de la simulation

    while env.peek() < simTime:
        global stop  # contient l'indice de la node concernée
        global colid  # true si le packet a subi une collision, false sinon
        if not stop == -1:  # pause dans la simulation : se fait à chaque fois q'un packet arrive a destination ou subi une collision
            if colid:
                nodes[stop].sf = random.randint(7, 12)
                pass

            #####################################
            if not colid:
                lEmitMoy.append(messStop.nbSend)
            #####################################

            tabSF = np.zeros(6)
            for node in nodes:
                tabSF[node.sf-7] += 1
            packetInSF.append(tabSF)

            stop = -1
            colid = False
        env.step()


def main():
    simTime = 1800000000  # Temps de simulation en ms (ici 500h)
    nbStation = 100  # Nombre de node dans le réseau
    period = 1800000  # Interval de temps moyen entre deux message (ici 30 min)
    packetLen = 20  # 20 octet dans le packet

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

    # ################################# affichage du nombre de station par SF
    plt.figure(figsize=(12, 8))
    tmp = np.array(packetInSF)
    plt.subplot(2, 2, 1)
    for i in range(6):
        st = "SF : " + str(i+7)
        plt.plot(tmp[:, i], label=st)
    plt.legend()
    plt.ylabel("nombre de nodes")

    # ################################ affichage du nombre d'émition des paquets
    plt.subplot(2, 2, 2)
    plt.plot(lEmitMoy)

    # ############################### affichage de l'emplacement des nodes et de la station
    axe = plt.subplot(2, 2, 3)
    plt.xlim([-300, 300])
    plt.ylim([-300, 300])
    coordlist = []
    for node in nodes:
        coordlist.append(node.coord)
    coordAray = np.array(coordlist)
    draw_circle = plt.Circle((0, 0), 200, fill=False)
    axe.set_aspect(1)
    axe.add_artist(draw_circle)

    plt.scatter(coordAray[:, 0], coordAray[:, 1], s=10)
    plt.scatter([0], [0], s=30)

    plt.show()
    print("end")

"""
Variables globales pour la simulation
"""

env = simpy.Environment()
packetsAtBS = []
nodes = []
station = []
stop = -1
colid = 0
lEmitMoy = []
packetInSF = []
messStop = None

# tableau des sensitivity par sf (en fonction du bandwidth)
sf7 = np.array([7, -126.5, -124.25, -120.75])
sf8 = np.array([8, -127.25, -126.75, -124.0])
sf9 = np.array([9, -131.25, -128.25, -127.5])
sf10 = np.array([10, -132.75, -130.25, -128.75])
sf11 = np.array([11, -134.5, -132.75, -128.75])
sf12 = np.array([12, -133.25, -132.25, -132.25])
sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

main()
