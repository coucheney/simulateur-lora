from math import floor

from Packet import Packet
from simu import Simu
import matplotlib.pyplot as plt
import numpy as np

smooth = 200


# comptage du nombres de nodes utilisant un SF
def nodePerSF(env: Simu, sf: int, nextSF: int, power: int, nextPower: int):
    if "stockSF" not in env.envData:
        env.addData([], "stockSF")
        env.addData([], "oldSF")
        env.addData([], "nodePerSF")
        env.addData([], "powSF")
        env.addData([], "powSFStock")

    if not env.envData["powSFStock"]: # premier comptage des SF
        firsCount = np.zeros((4, 6))
        for nd in env.envData["nodes"]:
            if nd.power <= 14:
                firsCount[0, nd.sf - 7] += 1
            elif 14 < nd.power <= 17:
                firsCount[1, nd.sf - 7] += 1
            else:
                firsCount[2, nd.sf - 7] += 1
            firsCount[3, nd.sf - 7] += 1
        env.envData["powSFStock"].append(firsCount)

    tmp = env.envData["powSFStock"][-1].copy()
    if power <= 14:
        tmp[0, sf - 7] -= 1
    elif 14 < power <= 17:
        tmp[1, sf - 7] -= 1
    else:
        tmp[2, sf - 7] -= 1
    if nextPower <= 14:
        tmp[0, nextSF - 7] += 1
    elif 14 < nextPower <= 17:
        tmp[1, nextSF - 7] += 1
    else:
        tmp[2, nextSF - 7] += 1
    tmp[3, sf - 7] -= 1
    tmp[3, nextSF - 7] += 1
    env.envData["powSFStock"].append(tmp)

    if len(env.envData["powSFStock"]) == smooth/4:
        tmp = sum(env.envData["powSFStock"])
        env.envData["powSF"].append(tmp/(smooth/4))
        env.envData["powSFStock"].pop(0)

# fonction qui collecte différentes données
def colectData(env: Simu, pack: Packet, time: float):
    if "timeOcc" not in env.envData:
        env.addData(np.zeros(len(env.envData["nodes"])), "timeOcc")
        env.addData([], "collidGraph")

    env.envData["timeOcc"][pack.nodeId] += pack.recTime

    #print(env.envData["send"], env.envData["collid"], env.envData["nbCapture"])
    colList = [env.envData["collid"] - env.envData["nbCapture"], env.envData["nbCapture"], env.envData["notHeard"]]
    env.envData["collidGraph"].append(colList)

    if "lastColid" not in env.envData:
        env.addData(0, "lastColid")
    if pack.lost and time > 10000000:
        env.envData["lastColid"] += 1

# calcul de la distance maximum en fonction des SF
def calcDistMax(sensi):
    maxDist = []
    for tab in sensi:
        temp = []
        for j in range(2, 22):
            temp.append(40 * 10 ** ((-j + 127.41 + tab[1]) / -20.8))
        maxDist.append(temp)
    return maxDist


# sauvegarde du nombre de fois ou un paquet à été émis
def nbReemit(env: Simu, pack: Packet):
    env.envData["reemit"].append(pack.nbSend)


# sauvegarde de l'énergie moyenne utilisé par un paquet
def colectMeanPower(env: Simu, pack: Packet):
    if "stockPower" not in env.envData:
        env.addData([], "stockPower")
        env.addData([], "averagePower")

    if not len(env.envData["stockPower"]) == smooth*2:
        env.envData["stockPower"].append(pack.energyCost)
    else:
        env.envData["averagePower"].append(sum(env.envData["stockPower"]) / (smooth*2))
        env.envData["stockPower"].pop(0)
        env.envData["stockPower"].append(pack.energyCost)

# création du graphique de la puissance moyenne
def drawMeanPower(env: Simu, axe):
    axe.set_title("Power by packet")
    axe.set_ylabel("power (milliampere-Heure)")
    axe.set_xlabel("packets sent")
    axe.set_ylim(0, max(env.envData["averagePower"]) + 0.01)
    axe.plot(env.envData["averagePower"])


# création du graphiques des nodes par SF
def drawNodePerSf(env: Simu, colors, fig, axes):
    fig.set_figheight(5)
    tmp = np.array(env.envData["powSF"])
    maxy = np.max(tmp[:, :3, :]) + 2
    for i in range(3):
        axes[0][i].set_ylim(0, maxy)
        axes[1][i].remove()
        axes[0][i].set_ylabel("nodes")
        axes[0][i].set_xlabel("packets sent")
    axes[0][0].set_title("low power (0-14 dB)")
    axes[0][1].set_title("med power (15-17 dB)")
    axes[0][2].set_title("high power (18-20 dB)")
    for i in range(6):
        axes[0][0].plot(tmp[:, 0, i], label="SF " + str(i + 7), c=colors[i])
    for i in range(6):
        axes[0][1].plot(tmp[:, 1, i], label="SF " + str(i + 7), c=colors[i])
    for i in range(6):
        axes[0][2].plot(tmp[:, 2, i], label="SF " + str(i + 7), c=colors[i])
    plt.legend(loc='upper center', bbox_to_anchor=(-0.73, -0.5), ncol=3)
    plt.savefig("graphic/sfPowerGraphic", dpi=400)
    plt.close()

    fig2, axes2 = plt.subplots(ncols=2, nrows=1, figsize=(18, 7))
    axes2[0].set_ylabel("nodes")
    axes2[0].set_xlabel("packets sent")
    axes2[0].set_title("Nodes per SF")
    for i in range(6):
        axes2[0].plot(tmp[:, 3, i], label="SF " + str(i + 7), c=colors[i])
    axes2[0].legend(ncol=3, bbox_to_anchor=(0.5, 1.15), loc='upper center')
    return fig2, axes2

# création du graphique qui affiche la position des nodes et la portée maximum des SF
def drawNodePosition(env: Simu, axe, colors):
    plt.title("Node placement")
    posx = [node.coord.x for node in env.envData["nodes"]]
    posy = [node.coord.y for node in env.envData["nodes"]]
    plt.scatter(posx, posy, s=10)
    maxDist = calcDistMax(env.envData["sensi"])
    maxLim = max(max(maxDist))
    plt.xlim(-maxLim - 20, maxLim + 20)
    plt.ylim(-maxLim - 20, maxLim + 20)
    coordBs = env.envData["BS"].coord
    plt.scatter(coordBs.x, coordBs.y, s=75, marker="^", c="red")
    for i in range(6):
        crc = plt.Circle((0, 0), maxDist[i][-1], fill=False, color=colors[i])
        axe[1].set_aspect(1)
        axe[1].add_artist(crc)


# création du graphique du nombre de réémition
def drawNbReemit(env: Simu, axes):
    tmp = env.envData["reemit"]
    axes.set_ylim(0, 7)
    axes.set_ylabel("re-emissions")
    axes.set_xlabel("packets sent")
    axes.set_title("Number of re-emissions")
    axes.scatter(range(len(tmp)), tmp, s=5, marker=".")

# création du graphique des collisions
def drawCollid(env):
    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
    leg = ["colide", "capture", "notHeard"]
    fmt = ["-", ":", "--"]
    tmp = np.array(env.envData["collidGraph"])
    for i in range(3):
        axe.plot(tmp[:, i], fmt[i], label=leg[i])
    axe.legend()
    axe.set_ylabel("number of collision")
    axe.set_xlabel("packets sent")
    axe.set_title("Collision")
    plt.savefig("graphic/collidGraphic", dpi=400)

# création des graphiques
def drawGraphics(env: Simu):
    colors = ["blue", "gold", "green", "red", "magenta", "black"]
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 9))

    fig2, axes2 = drawNodePerSf(env, colors, fig, axes)
    drawNodePosition(env, axes2, colors)
    plt.savefig("graphic/placement", dpi=400)
    plt.close()

    fig3, axes3 = plt.subplots(ncols=1, nrows=1, figsize=(9, 9))
    drawNbReemit(env, axes3)
    plt.savefig("graphic/reemit", dpi=400)
    plt.close()

    fig4, axes4 = plt.subplots(ncols=1, nrows=1, figsize=(9, 9))
    drawMeanPower(env, axes4)
    plt.savefig("graphic/power", dpi=400)

    drawCollid(env)

# fonction qui écrit les log des nodes
def getlog(env: Simu, nodeId: int, pack: Packet):
    if "log" not in env.envData:
        env.addData([], "log")
        for i in range(len(env.envData["nodes"])):
            env.envData["log"].append([])

    nd = env.envData["nodes"][nodeId]
    env.envData["log"][nodeId].append([env.simTime, floor(env.simTime / 86400000),nd.sf, nd.power, nd.battery.energyConsume, nd.firstSentPacket, nd.packetLost, nd.packetTotalLost])
