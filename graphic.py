from LoRa.Packet import Packet
from LoRa.func import calcDistMax
from LoRa.simu import Simu
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

def drawPacketPerSf(env: Simu):
    for i in range(6):
        plt.plot(env.envData["packetPerSF"])
    plt.show()
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

def drawNbReemit(env: Simu):
    tmp = env.envData["reemit"]
    plt.scatter(range(len(tmp)), tmp, s=5, marker=".")
    plt.show()
    print("0", tmp.count(0))
    print("1", tmp.count(1))
    print("2", tmp.count(2))
    print("3", tmp.count(3))
    print("4", tmp.count(4))

