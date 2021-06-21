from Packet import Packet
from simu import Simu
import matplotlib.pyplot as plt
import numpy as np


def nodePerSF(env: Simu, sf: int, nextSF: int):
    if env.envData["nodePerSF"]:
        tmp = list(env.envData["nodePerSF"][-1])
        tmp[sf-7] -= 1
        tmp[nextSF-7] += 1
        env.envData["nodePerSF"].append(tmp)
    else:
        tmp = [0, 0, 0, 0, 0, 0]
        for nd in env.envData["nodes"]:
            tmp[nd.sf-7] += 1
        env.envData["nodePerSF"].append(tmp)

def calcDistMax(sensi):
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 20):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    return maxDist

def nbReemit(env: Simu, pack: Packet):
    env.envData["reemit"].append(pack.nbSend)

def colectMeanPower(env: Simu, pack: Packet):
    if env.envData["averagePower"]:
        prevMean = env.envData["averagePower"][-1]
        average = (1 - (1 / len(env.envData["averagePower"]))) * prevMean + (1 / len(env.envData["averagePower"])) * pack.energyCost
        env.envData["averagePower"].append(average)
    else:
        env.envData["averagePower"].append(pack.energyCost)

def drawMeanPower(env: Simu):
    plt.ylim(0, max(env.envData["averagePower"])+0.02)
    plt.plot(env.envData["averagePower"])

def drawNodePerSf(env: Simu):
    tmp = np.array(env.envData["nodePerSF"])
    plt.ylabel("nodes")
    for i in range(6):
        plt.plot(tmp[:, i], label="SF " + str(i+7))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    plt.title("node per SF")

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
    drawNodePerSf(env)
    plt.subplot(2, 2, 2)
    drawNbReemit(env)
    axe = plt.subplot(2, 2, 3)
    drawNodePosition(env, axe)
    plt.subplot(2, 2, 4)
    drawMeanPower(env)
    plt.show()
