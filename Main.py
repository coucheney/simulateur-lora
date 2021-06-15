import math
import random

import numpy as np
from matplotlib import pyplot as plt
import learn
from BS import BS
from Event import sendPacketEvent, timmerEvent, mooveEvent, mooveDistEvent
from Node import Node
from Packet import Point
from graphic import drawGraphics, drawNodePosition
from simu import Simu


def readSensitivity():
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


readSensitivity()

# tableau des sensitivity par sf (en fonction du bandwidth)
# variable de LoRaSim
sensi = readSensitivity()
# tableau de distance maximum
TX = [22, 22, 22, 23,  # RFO/PA0: -2..1
      24, 24, 24, 25, 25, 25, 25, 26, 31, 32, 34, 35, 44,  # PA_BOOST/PA1: 2..14
      82, 85, 90,  # PA_BOOST/PA1: 15..17
      105, 115, 125]  # PA_BOOST/PA1+PA2: 18..20

s = Simu()
s.addData([], "nodes")
s.addData(sensi, "sensi")
s.addData(TX, "TX")
s.addData(200, "radius")
s.addData([], "nodePerSF")
s.addData(0, "send")
s.addData(0, "collid")
s.addData(BS(0, Point(0, 0)), "BS")
s.addData([], "reemit")
s.addData([], "averagePower")


def aleaPlacement(nbNode, radius):
    res = []
    for i in range(nbNode):
        a = random.random()
        b = random.random()
        if b < a:
            a, b = b, a
        posx = b * radius * math.cos(2 * math.pi * a / b)
        posy = b * radius * math.sin(2 * math.pi * a / b)
        res.append([posx, posy])
    return res


def gridPlacement(sizeGrid, radius):
    lin = np.linspace(-radius, radius, sizeGrid)
    res = []
    for i in range(sizeGrid):
        for j in range(sizeGrid):
            if not lin[i] == 0 or not lin[j] == 0:
                res.append([lin[i], lin[j]])
    return res


def linePlacement(nbNode, radius):
    lin = np.linspace(0, radius, nbNode + 1)
    res = []
    for i in lin[1:]:
        res.append([i, 0])
    return res


def evalParam(arg, settings):
    if arg[0] == "sf":
        if arg[1] == "rand" or (arg[1].isdigit() and 12 >= int(arg[1]) >= 7):
            settings["sf"] = arg[1]
        else:
            print("le paramètre sf doit être: rand, 7 ,8 ,9 ,10 ,11 ,12")
            exit()
    elif arg[0] in ["period", "packetLen", "radius"]:
        if arg[1].isdigit():
            settings[arg[0]] = int(arg[1])
        else:
            print("le paramètre period doit être un entiere postif")
            exit()
    elif arg[0] == "cr":
        if arg[1].isdigit() and 4 >= int(arg[1]) >= 1:
            settings["cr"] = int(arg[1])
        else:
            print("le paramètre period doit être un entiere postif compris entre 1 et 4")
            exit()
    elif arg[0] == "power":
        if arg[1].isdigit() and 20 >= int(arg[1]) >= 0:
            settings["power"] = int(arg[1])
        else:
            print("le paramètre period doit être un entier compris entre 0 et 20")
            exit()
    elif arg[0] == "algo":
        listAlgo = ["static", "rand", "TS", "TTTS", "EXP3", "UCB1"]
        if arg[1] in listAlgo:
            settings["algo"] = arg[1]
        else:
            print("l'algo", arg[1], "n'existe  pas")
            exit()
    else:
        print("l'argument", arg[0], "n'existe pas")
        exit()


def createAlgo(algo):
    if algo == "static":
        return learn.Static()
    elif algo == "TS":
        return learn.TS()
    elif algo == "TTTS":
        return learn.TTTS()
    elif algo == "EXP3":
        return learn.EXP3()
    elif algo == "UCB1":
        return learn.UCB1()
    else:
        return learn.RandChoise()


def loadNodeConfig():
    listFunc = [aleaPlacement, gridPlacement, linePlacement]
    listFuncArg = ["rand", "grid", "line"]
    id = 0
    with open("config/Nodes.txt", "r") as fi:
        lines = fi.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.rstrip()
            settings = {"period": 1800000, "packetLen": 20, "cr": 1, "sf": "rand", "power": 14, "radius": 200,
                        "algo": learn.RandChoise()}
            paramSplit = []
            param = line.split()
            for arg in param:
                paramSplit.append(arg.split(":"))

            print(paramSplit)
            for arg in paramSplit[2:]:
                evalParam(arg, settings)
            if not param[0].replace(".", "").replace("-", "").isdigit():
                try:
                    funcPlacement = listFunc[listFuncArg.index(param[0])]
                except ValueError:
                    print(param[0], "doit être : rand, grid, line")
                    exit()
                try:
                    place = funcPlacement(int(param[1]), int(settings["radius"]))
                except ValueError:
                    print(param[1], "doit être un entier")
                    exit()
            else:
                try:
                    place = [[float(param[0]), float(param[1])]]
                except ValueError:
                    print("les coordonées doivent être des float")
                    exit()

            for coord in place:
                if settings["sf"] == "rand":
                    sf = random.randint(7, 12)
                else:
                    sf = int(settings["sf"])
                algo = createAlgo(settings["algo"])
                s.envData["nodes"].append(
                    Node(id, settings["period"], s.envData["sensi"], s.envData["TX"], settings["packetLen"],
                         settings["cr"], 125, sf, settings["power"], Point(coord[0], coord[1]), settings["radius"],
                         algo))
                s.addEvent(sendPacketEvent(id, random.expovariate(1.0 / s.envData["nodes"][id].period), s, 0))
                id += 1


simTime = 1800000000
#simTime = 86400000    # 1 jours
s.addEvent(timmerEvent(0, s, simTime))

# mode de placement (pour le moment un seul possible en même temps)
loadNodeConfig()

# ########## Event scénario

s.addEvent(mooveDistEvent(100000, s, 500, 0))

###########

while s.simTime < simTime:
    s.nextEvent()

print("send :", s.envData["send"])
print("collid :", s.envData["collid"])

with open("res/saveENV.txt", "w") as fi:
    for nd in s.envData["nodes"]:
        algo = ["static", "rand"]
        if isinstance(nd.algo, learn.RandChoise):
            key = 1
        else:
            key = 0
        fi.write(str(nd.coord.x) + " " + str(nd.coord.y) + " sf:" + str(nd.sf) + " period:" + str(nd.period) +
                 " cr:" + str(nd.cr) + " packetLen:" + str(nd.packetLen) + " power:" + str(nd.power) + " algo:" + algo[
                     key] + "\n")

drawGraphics(s)
lowestBatterie = 0
for nd in s.envData["nodes"]:
    if lowestBatterie < nd.battery.energyConsume:
        lowestBatterie = nd.battery.energyConsume
print(lowestBatterie, "MiliAmpère-heure")
print("time: ", simTime / 86400000, "days")
