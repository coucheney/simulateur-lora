from LoRa.Packet import Packet
from LoRa.func import calcDistMax
from LoRa.simu import Simu
import matplotlib.pyplot as plt


def packetPerSF(env: Simu, SF, nextSF):
    if env.envData["packetPerSF"][0]:
        pass
    else:
        tmp = [0, 0, 0, 0, 0, 0]
        for nd in env.envData["nodes"]:
            tmp[nd.SF] += 1
        for i in range(6):
            env.envData["packetPerSF"][i] = tmp[i]

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

