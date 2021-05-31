
from LoRa.func import calcDistMax
from LoRa.simu import Simu
import matplotlib.pyplot as plt
import numpy as np


def packetPerSF(env: Simu, sf, nextSF):
    if env.envData["packetPerSF"]:
        tmp = env.envData["packetPerSF"][-1]
        tmp[sf-7] -= 1
        tmp[nextSF-7] += 1
        env.envData["packetPerSF"].append(tmp)
    else:
        tmp = [0, 0, 0, 0, 0, 0]
        for nd in env.envData["nodes"]:
            tmp[nd.sf-7] += 1
        env.envData["packetPerSF"].append(tmp)

def drawPacketPerSf(env: Simu):
    """
    for i in range(6):
        plt.plot(env.envData["packetPerSF"])
    plt.show()"""
    pass

def drawNodePosition(env: Simu):
    axe = plt.subplot()
    posx = [node.coord.x for node in env.envData["nodes"]]
    posy = [node.coord.y for node in env.envData["nodes"]]
    plt.scatter(posx, posy, s=10)
    plt.xlim(-750, 750)
    plt.ylim(-750, 750)
    maxDist = calcDistMax(env.envData["sensi"])
    for i in range(6):
        crc = plt.Circle((0, 0), maxDist[i][-1], fill=False)
        axe.set_aspect(1)
        axe.add_artist(crc)
    plt.show()

