import math
import random
from Packet import Packet, Point
from graphic import drawGraphics

# fonction qui renvoie des coordonée aléatoire dans un cercle
# radius : rayon dans lequel est placé le point
def aleaCoord(radius: int) -> Point:
    a = random.random()
    b = random.random()
    if b < a:
        a, b = b, a
    posx = b * radius * math.cos(2 * math.pi * a / b)
    posy = b * radius * math.sin(2 * math.pi * a / b)
    return Point(posx, posy)

# calcul des distance maximum en une node et l'antenne en fonction du SF et de la puissance
# sensi : tableau des sensibilité de l'antenne
def calcDistMax(sensi) -> list:
    # tableau de distance maximum
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 21):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    return maxDist

# Si les SF des deux paquets sont les mêmes, return True
def sfCollision(p1: Packet, p2: Packet) -> bool:
    if p1.sf == p2.sf:
        return True
    return False

"""
colision de fréquence entre les paquets 
la différence de fréquence minimal entre deux paquets est dans loraSim est de 6 dB
Le paquet 1 est celui qui arrive
Renvoie le ou les packet qui ont été perdu
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
        sim.envData["notHeard"] += 1

    if sim.envData["BS"].packetAtBS:  # au moins un paquet est en cours de réception
        for pack in sim.envData["BS"].packetAtBS:
            if sfCollision(packet, pack) and timingCollision(packet, pack, sim.simTime):
                packetColid = powerCollision(packet, pack)
                if len(packetColid) == 1:
                    sim.envData["nbCapture"] += 1
                for p in packetColid:
                    if not p.lost:
                        sim.envData["collid"] += 1
                    p.lost = True

# envoie d'un packet
# packet : paquet envoyé
# sendDate : temps d'envoie du packet
# sim : environement du simulateur
def send(packet: Packet, sendDate: float, sim) -> None:
    sim.envData["nodes"][packet.nodeId].packetSent += 1
    if packet.nbSend == 0:
        sim.envData["nodes"][packet.nodeId].firstSentPacket += 1
    packet.sendDate = sendDate
    collision(packet, sim)
    sim.envData["BS"].addPacket(packet)

# arrivée d'un packet dans l'antenne, il est ajouté si il ne subi pas de colision
# packet : packet reçu
# env : environement du simulateur
def packetArrived(packet: Packet, env) -> None:
    if packet in env.envData["BS"].packetAtBS:
        env.envData["BS"].removePacket(packet)
        env.envData["nodes"][packet.nodeId].setWaitTime(packet.recTime, env.simTime)
        battOk = env.envData["nodes"][packet.nodeId].battery.useEnergy(packet.energyCost)
        if not battOk:
            print(env.simTime / (24*60*60*1000), "days")
            drawGraphics(env)
            exit()
