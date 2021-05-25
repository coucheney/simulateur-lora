import math
import random
from LoRa.Packet import Packet, Point

"""
fonction qui renvoie des coordonée aléatoire dans un cercle 
"""
def aleaCoord(radius) -> Point:
    a = random.random()
    b = random.random()
    if b < a:
        a, b = b, a
    posx = b * radius * math.cos(2 * math.pi * a / b)
    posy = b * radius * math.sin(2 * math.pi * a / b)
    return Point(posx, posy)


def calcDistMax(sensi):
    # tableau de distance maximum
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 20):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    return maxDist

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


def timingCollision(p1: Packet, p2: Packet, simTime) -> bool:
    # assuming 8 preamble symbols
    Npream = 8
    # we can lose at most (Npream - 5) * Tsym of our preamble
    Tpreamb = 2 ** p1.sf / p1.bw * (Npream - 5)
    p2_end = p2.sendDate + p2.recTime
    p1_cs = simTime + Tpreamb
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


def collision(packet: Packet, sim):
    sensitivity = sim.envData["sensi"][packet.sf - 7, [125, 250, 500].index(packet.bw) + 1]
    if packet.rssi < sensitivity:  # La puissance du paquet est plus faible que la sensivitivity
        packet.lost = True

    if sim.envData["packetsAtBS"]:  # au moins un paquet est en cours de réception
        for pack in sim.envData["packetsAtBS"]:
            if sfCollision(packet, pack) and timingCollision(packet, pack, sim.simTime):
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


def send(packet: Packet, sendDate: float, sim) -> None:
    if sim.envData["nodes"][packet.nodeId].sendTime + sim.envData["nodes"][packet.nodeId].waitTime > sim.simTime:
        # print("---------", nodes[packet.nodeId].waitTime)
        #global contTmp
        #contTmp += 1
        pass
    sim.envData["nodes"][packet.nodeId].packetSent += 1
    packet.sendDate = sendDate
    collision(packet, sim)
    sim.envData["packetsAtBS"].append(packet)


"""
arrivée d'un packet dans l'antenne, il est ajouté si il ne subi pas de colision 
"""


def packetArrived(packet: Packet, env) -> None:
    if packet in env.envData["packetsAtBS"]:
        env.envData["packetsAtBS"].remove(packet)
        env.envData["nodes"][packet.nodeId].setWaitTime(packet.recTime, env.simTime)
