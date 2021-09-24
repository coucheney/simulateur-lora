from Battery import Battery
from func import calcDistMax, aleaCoord, Packet
from learn import *


class Node:
    """
    nodeid : id de la node
    period : temps moyen entre deux envoi de message (en ms)
    sf : spreading factor (entier compris entre 7 et 12)
    cr : coding rate (entier compris entre 1 et 4)
    bw : bandwith (125, 250 ou 500)
    coord : coordonées (x,y) de la node
    power : puissance d'émition des message en dB (entier compris entre -2 et 20 dB)
    packetLen : taille du packet
    freq : fréquence de la porteuse (860000000, 864000000, 868000000 hz)
            (peut valoir un entier entre 860000000 et 1020000000 par pas de 61hz, raport )
    packetSent : nombre de paquet envoyés
    firstSendPacket : compteur du nombre de première émition d'un paquet
    packetLost : compteur du nombre de packet perdu
    packetTotalLost : compteur du nombre de packet qui n'on jamais été reçu
    validCombination : liste contenant les combinaison de paramètre valide
    waitTime : temps pendant lequel le noeud doit attendre au minimum avant de réémètre
    sendTime : temps ou le noeud à pour la dernière fois émis
    TX : tableau des rapport puisance et milliAmpère-Heure
    algo : objet corespondant a l'algorithme permètant les choix SF / puissance
    active : booléen signifiant si le noeud emmet ou non
    battery : objet corespondant à la batterie du noeud
    waitPacket : liste des packet en attente /! pas utilisé
    """

    def __init__(self, nodeId: int, period: int, sensi, TX, packetLen=20, cr=1, bw=125, sf=7, power=14, coord=None,
                 radius=200, algo=Static()):
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
        self.firstSentPacket = 0
        self.packetLost = 0
        self.packetTotalLost = 0
        self.validCombination = self.checkCombination(sensi)
        self.waitTime = 0
        self.sendTime = 0
        self.TX = TX
        self.algo = algo
        self.algo.start(n_arms=len(self.validCombination))
        if isinstance(self.algo, Static) or isinstance(self.algo, RandChoise):
            self.sf = sf
            self.power = power
        self.active = False
        self.battery = Battery(10000000)
        self.waitPacket = []

    # construction de la liste contenant les combinaisons de paramètre valide (SF + Power)
    # sensi : tableau des sensibilité de l'antenne
    def checkCombination(self, sensi):
        lTemp = []
        maxDist = calcDistMax(sensi)
        for i in range(len(maxDist)):
            for j in range(len(maxDist[i])):
                if maxDist[i][j] > math.sqrt((self.coord.x - 0) ** 2 + (self.coord.y - 0) ** 2):
                    lTemp.append([i + 7, j + 2])
        return lTemp

    def __str__(self):
        return "node : {:<4d} sf : {:<3d} packetSent : {:<6d} packetLost : {:<6d} messageLost : {:<6d} power : {:<3d}".format(
            self.nodeId, self.sf, self.packetSent, self.packetLost, self.packetTotalLost, self.power)

    # Création d'un paquet corespondant aux paramètres de la node
    def createPacket(self, idPacket) -> Packet:
        p = Packet(self.nodeId, self.packetLen, self.sf, self.cr, self.bw, self.coord, self.power, self.TX, idPacket, self.freq)
        return p

    # définit le temps d'attente minimum avant le prochain envoie (99 fois le temps d'envoie)
    # time : temps ou le noeud à précédament été utilisé
    # sendTime : temps actuelle de la simulation
    def setWaitTime(self, time: float, sendTime: float):
        self.waitTime = 99 * time
        self.sendTime = sendTime

    def readSensitivity(self):
        try:
            sensi = []
            with open("config/sensitivity.txt", "r") as fi:
                lines = fi.readlines()
                for line in lines:
                    line = line.split(" ")
                    tmp = []
                    for val in line:
                        if val:
                            tmp.append(val)
                    tmp = [float(val) for val in tmp]
                    sensi.append(np.array(tmp))
            return np.array(sensi)
        except ValueError:
            print("erreur dans sensitivity.txt")
            exit()

    """
        Fonction qui crée un vecteur de stratégies autour d'un couple (SF, PW)
    """
    def set_parameter4(self, sf, power):
        if power == 20 and 7 < sf < 12:
            list = [[sf - 1, power], [sf, power], [sf + 1, power - 2], [sf + 1, power - 1], [sf + 1, power]]
        elif power == 20 and sf == 7:
            list = [[sf, power], [sf + 1, power - 2], [sf + 1, power - 1], [sf + 1, power]]
        elif sf == 7:
            list = [[7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [7, 10], [7, 11],
                    [7, 12], [7, 13], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18],
                    [7, 19]]
        elif power == 0:
            list = [[sf, power], [sf, power + 1], [sf, power + 2], [sf, power + 3], [sf, power + 4]]
        elif 8 < sf < 12 and power < 18:
            list = [[sf - 2, power + 2], [sf - 1, power + 1], [sf, power], [sf + 1, power - 2], [sf + 1, power - 1],
                    [sf + 1, power]]
        elif 8 < sf < 12 and power < 18:
            list = [[sf - 2, power + 2], [sf - 1, power + 1], [sf, power], [sf + 1, power - 2], [sf + 1, power - 1],
                    [sf + 1, power]]
        elif 8 < sf < 12 and power >= 18:
            list = [[sf - 2, 20], [sf - 1, 20], [sf - 1, 19], [sf, power], [sf + 1, power - 2], [sf + 1, power - 1],
                    [sf + 1, power]]
        else:
            list = [[sf, power]]
        self.validCombination = list
        self.algo.start(n_arms=len(self.validCombination))


    """
        Fonction qui retourne un vecteur avec tous les couples (SF, Power) possibles. Les puissances 2,3,5,6,7 sont retirés du fait qu'ils ne font que diminuer la puissance,
        mais n'apportent aucun gains en terme de consommation énergétique.
    """
    def checkCombination2(self, sensi=[]):
        sf = [7, 8, 9, 10, 11, 12]
        pw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        combi = []
        for i in sf:
            for j in pw:
                combi.append([i, j])
        deleteList = [2, 3, 5, 6, 7]
        toDelete = []
        for elem in combi:
            if elem[1] in deleteList:
                toDelete.append(elem)
        for elem in toDelete:
            combi.remove(elem)
        return combi

    def set_parameter(self, sf, power):
        self.validCombination[0] = [sf, power]
        self.algo.index = 0
