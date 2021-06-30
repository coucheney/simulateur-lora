from Packet import Packet
from simu import Simu
import matplotlib.pyplot as plt
import numpy as np


# comptage du nombres de nodes utilisant un SF
def nodePerSF(env: Simu, sf: int, nextSF: int):
    if "stockSF" not in env.envData:
        env.addData([], "stockSF")
        env.addData([], "oldSF")
        env.addData([], "nodePerSF")

    if env.envData["oldSF"]:
        tmp = env.envData["oldSF"]
        tmp[sf - 7] -= 1
        tmp[nextSF - 7] += 1
        env.envData["stockSF"].append(tmp)
    else:
        tmp = [0, 0, 0, 0, 0, 0]
        for nd in env.envData["nodes"]:
            tmp[nd.sf - 7] += 1
        env.envData["oldSF"] = tmp

    if len(env.envData["stockSF"]) < 10:
        env.envData["stockSF"].append(tmp)
    else:
        ltmp = np.zeros(6)
        for tab in env.envData["stockSF"]:
            ltmp += np.array(tab)
        env.envData["nodePerSF"].append(ltmp / 10)
        env.envData["stockSF"] = []

def colectData(env: Simu, pack: Packet):
    if "colidSfPower" not in env.envData:
        env.addData(np.zeros((6, 21)), "colidSfPower")
        env.addData(np.zeros(len(env.envData["nodes"])), "timeOcc")

    if pack.lost:
        env.envData["colidSfPower"][pack.sf-7, pack.power] += 1

    env.envData["timeOcc"][pack.nodeId] += pack.recTime

# calcul de la distance maximum en fonction des SF
def calcDistMax(sensi):
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 20):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    print(maxDist)
    return maxDist


# sauvegarde du nombre de fois ou un paquet à été émis
def nbReemit(env: Simu, pack: Packet):
    env.envData["reemit"].append(pack.nbSend)


# sauvegarde de l'énergie moyenne utilisé par un paquet
def colectMeanPower(env: Simu, pack: Packet):
    if "stockPower" not in env.envData:
        env.addData([], "stockPower")
        env.addData([], "averagePower")

    if not len(env.envData["stockPower"]) == 10:
        env.envData["stockPower"].append(pack.energyCost)
    else:
        env.envData["averagePower"].append(sum(env.envData["stockPower"]) / 10)
        env.envData["stockPower"] = []


# création du graphique de la puissance moyenne
def drawMeanPower(env: Simu):
    plt.ylim(0, max(env.envData["averagePower"]) + 0.02)
    plt.plot(env.envData["averagePower"])


# création du graphiques des nodes par SF
def drawNodePerSf(env: Simu):
    tmp = np.array(env.envData["nodePerSF"])
    plt.ylabel("nodes")
    for i in range(6):
        plt.plot(tmp[:, i], label="SF " + str(i + 7))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    plt.title("node per SF")


# création du graphique qui affiche la position des nodes et la portée maximum des SF
def drawNodePosition(env: Simu, axe):
    plt.title("Node placement")
    posx = [node.coord.x for node in env.envData["nodes"]]
    posy = [node.coord.y for node in env.envData["nodes"]]
    plt.scatter(posx, posy, s=10)
    maxDist = calcDistMax(env.envData["sensi"])
    maxLim = max(max(maxDist))
    plt.xlim(-maxLim - 20, maxLim + 20)
    plt.ylim(-maxLim - 20, maxLim + 20)
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    coordBs = env.envData["BS"].coord
    plt.scatter(coordBs.x, coordBs.y, s=14, marker="^")
    for i in range(6):
        crc = plt.Circle((0, 0), maxDist[i][-1], fill=False, color=colors[i])
        axe.set_aspect(1)
        axe.add_artist(crc)


# création du graphique du nombre de réémition
def drawNbReemit(env: Simu):
    tmp = env.envData["reemit"]
    plt.ylim(0, 7)
    plt.ylabel("re-emissions")
    plt.title("Number of re-emissions")
    plt.scatter(range(len(tmp)), tmp, s=5, marker=".")


# affichage des graphiques
def drawGraphics(env: Simu):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    drawNodePerSf(env)
    plt.subplot(2, 2, 2)
    drawNbReemit(env)
    axe = plt.subplot(2, 2, 3)
    drawNodePosition(env, axe)
    plt.subplot(2, 2, 4)
    drawMeanPower(env)
    plt.show()

# fonction qui écrit les log des nodes
def getlog(env: Simu, nodeId: int, pack: Packet):
    if "log" not in env.envData:
        env.addData([], "log")
        for i in range(len(env.envData["nodes"])):
            env.envData["log"].append([])

    nd = env.envData["nodes"][nodeId]
    env.envData["log"][nodeId].append([nd.sf, nd.power, nd.battery.energyConsume, nd.firstSentPacket, nd.packetLost, nd.packetTotalLost])
