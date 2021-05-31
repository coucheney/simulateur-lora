import math
import random
import simpy
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, nodeId: int, packetLen: int, sf: int, cr: int, bw: int, pNode: Point, power: int):
        self.nodeId = nodeId
        self.packetLen = packetLen
        self.sf = sf
        self.cr = cr
        self.bw = bw
        self.recTime = self.airTime()  # temps de transmition
        self.lost = False
        self.nbSend = 0
        self.rssi = calcRssi(pNode, Point(0, 0), power)  # [0, 0] pour le moment l'antenne est placée en 0,0
        self.sendDate = None
        self.freq = random.choice([860000000, 864000000, 868000000])
        self.power = power
        self.energyCost = (self.recTime * TX[int(self.power) + 2] * 3) / 1e6

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
        #print(Tpaquet, self.sf)
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


"""
fonction qui renvoie des coordonée aléatoire dans un cercle 
"""


def aleaCoord() -> Point:
    a = random.random()
    b = random.random()
    global radius
    if b < a:
        a, b = b, a
    posx = b * radius * math.cos(2 * math.pi * a / b)
    posy = b * radius * math.sin(2 * math.pi * a / b)
    return Point(posx, posy)


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

    def __init__(self, nodeId: int, period: int, packetLen=20, cr=1, bw=125, sf=7, power=14, coord=None):
        if coord is None:
            coord = aleaCoord()
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
        self.validCombination = self.checkCombination()
        self.waitTime = 0
        self.sendTime = 0

    """
    construction de la liste contenant les combinaisons de paramètre valide (SF + Power)
    """

    def checkCombination(self) -> list:
        lTemp = []
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

    def createPacket(self) -> Packet:
        p = Packet(self.nodeId, self.packetLen, self.sf, self.cr, self.bw, self.coord, self.power)
        return p

    def setWaitTime(self, time: float, sendTime: float):
        self.waitTime = 99 * time
        self.sendTime = sendTime


"""
Si les SF des deux paquets sont les mêmes, return True
"""


def sfCollision(p1: Packet, p2: Packet) -> bool:
    if p1.sf == p2.sf:
        return True
    return False

# ####################################################################### Pas utilisé questionnement en cour
# frequencyCollision, conditions
#
#        |f1-f2| <= 120 kHz if f1 or f2 has bw 500
#        |f1-f2| <= 60 kHz if f1 or f2 has bw 250
#        |f1-f2| <= 30 kHz if f1 or f2 has bw 125
#   fonction utilisée entre deux paquet sur le même SF
#   avec les paramètres 868.10, 868.30 et 868.50, il ne peut pas y avoir de collision si p1.freq != p2.freq
def frequencyCollision(p1: Packet, p2: Packet) -> bool:
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


def powerCollision(p1: Packet, p2: Packet) -> list:
    powerThreshold = 6  # dB
    if abs(
            p1.rssi - p2.rssi) < powerThreshold:  # La puissance des deux message est trop proche, les deux paquet sont perdu
        return [p1, p2]
    elif p1.rssi < p2.rssi:  # le paquet qui arrive est moins puissant que le paquet déja présent, le paquet qui arrive est perdu
        return [p1]
    else:  # le paquet qui vient d'arriver est plus puissant
        return [p2]


"""
collision au niveau des timming des paquets
retourne False si au moins 5 symbole du preambule du paquet 1, sont emis après la fin du paquet 2
True sinon (collision entre les deux paquets) 
"""


def timingCollision(p1: Packet, p2: Packet) -> bool:
    # assuming 8 preamble symbols
    Npream = 8
    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2 ** p1.sf / p1.bw * (Npream - 5)
    p2_end = p2.sendDate + p2.recTime
    p1_cs = env.now + Tpreamb
    if p1_cs < p2_end:  # il faut que l'antenne puisse capter au moins les 5 derniers symbole du préambule
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


def collision(packet: Packet):
    sensitivity = sensi[packet.sf - 7, [125, 250, 500].index(packet.bw) + 1]
    if packet.rssi < sensitivity:  # La puissance du paquet est plus faible que la sensivitivity
        packet.lost = True

    if packetsAtBS:  # au moins un paquet est en cours de réception
        for pack in packetsAtBS:
            if sfCollision(packet, pack) and timingCollision(packet, pack):
                packetColid = powerCollision(packet, pack)
                for p in packetColid:
                    p.lost = True
            """if interSFCollision(packet, pack):
                packet.lost = True
                pack.lost = True"""


"""
envoie d'un packet 
Si pas de collision => le paquet est ajouté a la liste des paquets qui arrive
"""


def send(packet: Packet, sendDate: float) -> None:
    if nodes[packet.nodeId].sendTime + nodes[packet.nodeId].waitTime > env.now:
        # print("---------", nodes[packet.nodeId].waitTime)
        global contTmp
        contTmp += 1
    nodes[packet.nodeId].packetSent += 1
    packet.sendDate = sendDate
    collision(packet)
    packetsAtBS.append(packet)


"""
arrivée d'un packet dans l'antenne, il est ajouté si il ne subi pas de colision 
"""


def packetArrived(packet: Packet) -> None:
    if packet in packetsAtBS:
        global messStop
        messStop = packet
        packetsAtBS.remove(packet)
        nodes[packet.nodeId].setWaitTime(packet.recTime, env.now)


"""
prosessus de transmition d'un paquet
"""


def transmit(packet: Packet, period: float):
    time = random.expovariate(1.0 / float(period))
    yield env.timeout(time)  # date de début de l'envoie du packet
    # print(time)
    send(packet, env.now)  # le packet est envoyé
    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    global stop
    global colid
    global messStop
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        stop = packet.nodeId
        colid = True
        messStop = packet
        newPacket = nodes[packet.nodeId].createPacket()
        newPacket.nbSend += 1
        env.process(reTransmit(newPacket))
    else:
        stop = packet.nodeId
        colid = False
    packetArrived(packet)
    env.process(transmit(nodes[packet.nodeId].createPacket(), period))



"""
processus qui permet le retransmition d'un paquet, si celui ci a subi une colision
"""
total = 0
cont = 0
def reTransmit(packet: Packet):
    time = random.expovariate(1 / (packet.recTime * packet.nbSend))
    #time = packet.recTime * packet.nbSend
    yield env.timeout(time)  # date de début de l'envoie du packet
    send(packet, env.now)  # le packet est envoyé

    yield env.timeout(packet.recTime)  # temps ou le signal est émis
    # le temps de reception est passé
    global stop
    global colid
    global messStop
    if packet.lost:
        nodes[packet.nodeId].packetLost += 1
        if packet.nbSend < 7:
            stop = packet.nodeId
            colid = True
            messStop = packet
            newPacket = nodes[packet.nodeId].createPacket()
            newPacket.nbSend = packet.nbSend + 1
            env.process(reTransmit(newPacket))
        else:
            nodes[packet.nodeId].messageLost += 1
    else:
        packetArrived(packet)
        stop = packet.nodeId
        colid = False


def affTime():
    yield env.timeout(1800000000 / 10)
    print((env.now / 1800000000) * 100, "%")
    env.process(affTime())


"""
début de la simulation 
"""


def startSimulation(simTime: float, nbStation: int, period: int, packetLen: int, graphic: bool) -> None:
    # création des nodes, des paquets et des évent d'envoi des messages
    """"""
    for idStation in range(nbStation):
        node = Node(idStation, period, packetLen, sf=random.randint(7, 12))
        # node = Node(idStation, period, packetLen, sf=7, bw=125, cr=4)
        nodes.append(node)
        env.process(transmit(node.createPacket(), node.period))
    """"""
    """
    sizeRepart = int(nbStation / 10)
    for i in range(0, sizeRepart):
        for j in range(0, sizeRepart):
            node = Node(i + j, period, packetLen, sf=random.randint(7, 12), coord=Point(((radius*2)/10) * i - radius, ((radius*2)/10) * j - radius))
            if not (node.coord.x == 0 and node.coord.y == 0):
                print(((radius*2)/sizeRepart) * i - radius, ((radius*2)/sizeRepart) * j - radius)
                nodes.append(node)
                env.process(transmit(node.createPacket(), node.period))
    """
    env.process(affTime())
    # lancement de la simulation


    while env.peek() < simTime:
        global stop  # contient l'indice de la node concernée
        global colid  # true si le packet a subi une collision, false sinon
        if not stop == -1:  # pause dans la simulation : se fait à chaque fois q'un packet arrive a destination ou subi une collision
            if colid:
                tmp = random.choice(nodes[stop].validCombination)
                nodes[stop].sf = tmp[0]
                nodes[stop].power = tmp[1]
                pass
            if graphic:
                dataGraphic()

            stop = -1
            colid = False
        env.step()


"""
fonction qui colecte les données pour les graphique
"""


def dataGraphic() -> None:
    global colid
    if colid:
        lEmitMoy.append(messStop.nbSend)
    tabSFtemp = np.zeros(6)
    tabSF = np.zeros(6)
    for node in nodes:
        tabSFtemp[node.sf - 7] += 1

    if packetInSF:
        acttabSF = packetInSF[-1]
        for i in range(6):
            tabSF[i] = (1 - (1 / len(packetInSF))) * acttabSF[i] + (1 / len(packetInSF)) * tabSFtemp[i]
            # tabSF[i] = tabSFtemp[i]
    packetInSF.append(tabSF)

    id = messStop.nodeId
    consE = messStop.energyCost
    if len(powerListGraphic[id]) == 0:
        powerListGraphic[id].append(consE)
    else:
        # print(((1-(1/(len(powerListGraphic)+1))) * powerListGraphic[-1]))
        # print(((1 / (len(powerListGraphic)+1)) * consE))
        powerListGraphic[id].append(((1 - (1 / (len(powerListGraphic[id]) + 1))) * powerListGraphic[id][-1]) + (
                (1 / (len(powerListGraphic[id]) + 1)) * consE))
        # powerListGraphic.append(consE)
        # print("-----------")

    if len(powerListGraphic[-1]) == 0:
        powerListGraphic[-1].append(consE)
    else:
        powerListGraphic[-1].append(((1 - (1 / (len(powerListGraphic[-1]) + 1))) * powerListGraphic[-1][-1]) + (
                (1 / (len(powerListGraphic[-1]) + 1)) * consE))

    global sumPowerNetwork
    sumPowerNetwork += messStop.energyCost


"""
fonction qui dessine les grahique 
"""


def drawGraphic() -> None:
    # ################################# affichage du nombre de station par SF
    plt.figure(figsize=(12, 8))
    tmp = np.array(packetInSF)
    plt.subplot(2, 2, 1)
    for i in range(6):
        st = "SF : " + str(i + 7)
        plt.plot(tmp[1:, i], label=st)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3)
    plt.ylim(0)
    plt.ylabel("nombre de nodes")
    plt.pause(0.1)

    # ################################ affichage du nombre d'émition des paquets
    plt.subplot(2, 2, 2)
    plt.ylim(0, 8)
    plt.scatter(range(len(lEmitMoy)), lEmitMoy, s=5, marker=".")

    # ################################ affichage de l'emplacement des nodes et de la station
    axe = plt.subplot(2, 2, 3)
    global radius
    plt.xlim([-radius - 30, radius + 30])
    plt.ylim([-radius - 30, radius + 30])
    # draw_circle = plt.Circle((0, 0), radius, fill=False)

    for i in range(6):
        crc = plt.Circle((0, 0), maxDist[i][-1], fill=False)
        axe.set_aspect(1)
        axe.add_artist(crc)
    plt.scatter([node.coord.x for node in nodes], [node.coord.y for node in nodes], s=10)
    plt.scatter([0], [0], s=30)

    # ############################### affichage de la consomation
    plt.subplot(2, 2, 4)
    plt.ylim([0, max(powerListGraphic[-1]) + 0.1])
    # for i in range(len(nodes)):
    #    plt.plot(powerListGraphic[i])
    plt.plot(powerListGraphic[-1])

    plt.show()


def main(graphic=False) -> float:
    simTime = 1800000000 # Temps de simulation en ms (ici 500h)
    nbStation = 100  # Nombre de node dans le réseau
    period = 1800000  # Interval de temps moyen entre deux message (ici 30 min)
    packetLen = 20  # 20 octet dans le packet

    for i in range(nbStation + 1):
        powerListGraphic.append([])

    # lancement de la simulation
    startSimulation(simTime, nbStation, period, packetLen, graphic)

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
    print("percent lost :", (sumLost / sumSend) * 100, "%")

    if graphic:
        drawGraphic()
    return sumLost / sumSend


"""
fonction qui calcule le tableau de distance maximum entre les nodes et l'antenne, en fonction du SF et de la puissance
"""


def calcDistMax():
    # tableau de distance maximum
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 20):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    print(np.array(maxDist))
    return maxDist


"""
Variables globales pour la simulation
"""
# tableau des sensitivity par sf (en fonction du bandwidth)
# variable de LoRaSim
sf7 = np.array([7, -126.5, -124.25, -120.75])
sf8 = np.array([8, -127.25, -126.75, -124.0])
sf9 = np.array([9, -131.25, -128.25, -127.5])
sf10 = np.array([10, -132.75, -130.25, -128.75])
sf11 = np.array([11, -134.5, -132.75, -128.75])
sf12 = np.array([12, -133.25, -132.25, -132.25])
sensi = np.array([sf7, sf8, sf9, sf10, sf11, sf12])

# consomation en mA, en fonction du la puissance en dB
TX = [22, 22, 22, 23,  # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,  # PA_BOOST/PA1: 15..17
      105, 115, 125]  # PA_BOOST/PA1+PA2: 18..20

# tableau de distance maximum
maxDist = calcDistMax()

# variable utilisées pour la simulation
env = simpy.Environment()
packetsAtBS = []
nodes = []
station = []
stop = -1
colid = 0
lEmitMoy = []
packetInSF = []
powerListGraphic = []
messStop = None
radius = 200
sumPowerNetwork = 0
contTmp = 0
main()
print("total :", sumPowerNetwork)
print("EDC", contTmp)
