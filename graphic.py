from Packet import Packet
from func import calcDistMax
from simu import Simu
import matplotlib.pyplot as plt
import numpy as np


def packetPerSF(env: Simu, sf, nextSF):
    if env.envData["packetPerSF"]:
        tmp = list(env.envData["packetPerSF"][-1])
        tmp[sf-7] -= 1
        tmp[nextSF-7] += 1
        env.envData["packetPerSF"].append(tmp)
    else:
        tmp = [0, 0, 0, 0, 0, 0]
        for nd in env.envData["nodes"]:
            tmp[nd.sf-7] += 1
        env.envData["packetPerSF"].append(tmp)

def nbReemit(env: Simu, pack: Packet):
    env.envData["reemit"].append(pack.nbSend)

def colectMeanPower(env: Simu, pack: Packet):
    if env.envData["averagePower"]:
        prevMean = env.envData["averagePower"][-1]
        average = (1 - (1 / len(env.envData["averagePower"]))) * prevMean + (1 / len(env.envData["averagePower"])) * pack.power
        env.envData["averagePower"].append(average)
    else:
        env.envData["averagePower"].append(pack.power)

def drawMeanPower(env: Simu):
    plt.plot(env.envData["averagePower"])

def drawPacketPerSf(env: Simu):
    tmp = np.array(env.envData["packetPerSF"])
    plt.ylabel("nodes")
    for i in range(6):
        plt.plot(tmp[:, i], label="SF " + str(i+7))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    plt.title("packet per SF")

def drawNodePosition(env: Simu, axe):
    plt.title("Node placement")
    posx = [node.coord.x for node in env.envData["nodes"]]
    posy = [node.coord.y for node in env.envData["nodes"]]
    plt.scatter(posx, posy, s=10)
    maxDist = calcDistMax(env.envData["sensi"])
    maxLim = max(max(maxDist))
    plt.xlim(-maxLim-20, maxLim+20)
    plt.ylim(-maxLim-20, maxLim+20)
    colors = ["blue", "orange", "green", "red", "purple", "brown"]
    coordBs = env.envData["BS"].coord
    plt.scatter(coordBs.x, coordBs.y, s=14, marker="^")
    for i in range(6):
        crc = plt.Circle((0, 0), maxDist[i][-1], fill=False, color=colors[i])
        axe.set_aspect(1)
        axe.add_artist(crc)

def drawNbReemit(env: Simu):
    tmp = env.envData["reemit"]
    plt.ylim(0, 7)
    plt.ylabel("re-emissions")
    plt.title("Number of re-emissions")
    plt.scatter(range(len(tmp)), tmp, s=5, marker=".")

def drawGraphics(env: Simu):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    drawPacketPerSf(env)
    plt.subplot(2, 2, 2)
    drawNbReemit(env)
    axe = plt.subplot(2, 2, 3)
    drawNodePosition(env, axe)
    plt.subplot(2, 2, 4)
    drawMeanPower(env)
    plt.show()
