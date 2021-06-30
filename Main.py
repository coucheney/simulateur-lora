import math
import random
import numpy as np
import learn
from BS import BS
from Event import sendPacketEvent, timmerEvent, mooveDistEvent
from Node import Node
from Packet import Point
from graphic import drawGraphics
from simu import Simu


# lecture du fichier de configuration de la sensibilité de l'entenne
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


# placement aléatoire des nodes sur un disque de rayon radius
def aleaPlacement(nbNode: int, radius: float):
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


# placement des nodes sur une grille
def gridPlacement(sizeGrid: int, radius: float):
    lin = np.linspace(-radius, radius, sizeGrid)
    res = []
    for i in range(sizeGrid):
        for j in range(sizeGrid):
            if not lin[i] == 0 or not lin[j] == 0:
                res.append([lin[i], lin[j]])
    return res


# placement des nodes sur une ligne
def linePlacement(nbNode: int, radius: float):
    lin = np.linspace(0, radius, nbNode + 1)
    res = []
    for i in lin[1:]:
        res.append([i, 0])
    return res


# evaluation des paramètres des nodes
def evalParam(arg, settings, listAlgo):
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
        if arg[1] in listAlgo:
            settings["algo"] = arg[1]
        else:
            print("l'algo", arg[1], "n'existe  pas")
            exit()
    else:
        print("l'argument", arg[0], "n'existe pas")
        exit()


# creation de l'objec algo
def createAlgo(algo, listAlgo, listObjAlgo):
    tmp = listAlgo.index(algo)
    return eval(listObjAlgo[tmp])


# lecture du fichier de configuration des algo de choix de paramètres (SF/Power)
def readConfigAlgo():
    listAlgo = []
    listObjAlgo = []
    with open("config/configAlgo.txt", "r") as fi:
        lines = fi.readlines()
        for line in lines:
            tmp = line.split()
            listAlgo.append(tmp[0])
            listObjAlgo.append(tmp[1])
    return listAlgo, listObjAlgo


# création de la node en fonction des paramètres
def placeNode(settings, listAlgo, listObjAlgo, coord, nodeId, env):
    if settings["sf"] == "rand":
        sf = random.randint(7, 12)
    else:
        sf = int(settings["sf"])
    algo = createAlgo(settings["algo"], listAlgo, listObjAlgo)
    env.envData["nodes"].append(
        Node(nodeId, settings["period"], env.envData["sensi"], env.envData["TX"], settings["packetLen"],
             settings["cr"], 125, sf, settings["power"], Point(coord[0], coord[1]), settings["radius"],
             algo))
    env.addEvent(sendPacketEvent(nodeId, random.expovariate(1.0 / env.envData["nodes"][nodeId].period), env, 0))


# placement des nodes à partir du fichier de configuration
def loadNodeConfig(env):
    listAlgo, listObjAlgo = readConfigAlgo()
    listFunc = [aleaPlacement, gridPlacement, linePlacement]
    listFuncArg = ["rand", "grid", "line"]
    nodeId = 0
    with open("config/Nodes.txt", "r") as fi:
        lines = fi.readlines()
        for line in lines:
            line = line.replace("\n", "")
            line = line.rstrip()
            settings = {"period": 1800000, "packetLen": 20, "cr": 1, "sf": "rand", "power": 14, "radius": 200,
                        "algo": learn.RandChoise()}
            paramSplit = []
            param = line.split()
            print("---", param)
            for arg in param:
                paramSplit.append(arg.split(":"))

            print(paramSplit)
            for arg in paramSplit[2:]:
                evalParam(arg, settings, listAlgo)
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
                placeNode(settings, listAlgo, listObjAlgo, coord, nodeId, env)
                nodeId += 1


# fonction qui sauvegarde la configuration des nodes de la simulation
def saveConfig(env):
    with open("config/saveENV.txt", "w") as fi:
        for nd in env.envData["nodes"]:
            algo = ["static", "rand"]
            if isinstance(nd.algo, learn.RandChoise):
                key = 1
            else:
                key = 0
            fi.write(str(nd.coord.x) + " " + str(nd.coord.y) + " sf:" + str(nd.sf) + " period:" + str(nd.period) +
                     " cr:" + str(nd.cr) + " packetLen:" + str(nd.packetLen) + " power:" + str(nd.power) + " algo:" +
                     algo[
                         key] + "\n")


# chargement du tableau TX (consommation en mA en fonction de la puissance)
# pour le moment le tableau doit couvrir toutes les puissance entre -2 et 20
def loadTX():
    with open("config/TX.txt", "r") as fi:
        line = fi.readline()
        line = line.split()
        if not len(line) == 23:
            print("TX n'a pas le bon nombre d'argument")
            exit()
        return [int(val) for val in line]


# création de l'objet simulation et ajout des paramètre
def initSimulation():
    s = Simu()
    s.addData([], "nodes")
    s.addData(readSensitivity(), "sensi")
    s.addData(loadTX(), "TX")
    s.addData(200, "radius")
    s.addData([], "nodePerSF")
    s.addData(0, "send")
    s.addData(0, "collid")
    s.addData(BS(0, Point(0, 0)), "BS")
    s.addData([], "reemit")
    s.addData(0, "nbCapture")
    s.addData(0, "notHeard")
    return s


def main():
    s = initSimulation()
    # simTime = 1800000000   # temp de l'article
    simTime = 86400000 * 10  # 1 jours
    s.addEvent(timmerEvent(0, s, simTime, 0))
    loadNodeConfig(s)

    # ########## Event scénario

    # s.addEvent(mooveDistEvent(100000, s, 500, 0))

    ###########

    while s.simTime < simTime:
        s.nextEvent()

    print("send :", s.envData["send"])
    print("collid :", s.envData["collid"])
    saveConfig(s)
    lowestBatterie = 0
    # tm = 0
    for nd in s.envData["nodes"]:
        if lowestBatterie < nd.battery.energyConsume:
            lowestBatterie = nd.battery.energyConsume
            # tm = nd.nodeId
    # print(tm)
    print(lowestBatterie, "MiliAmpère-heure")
    print("time: ", simTime / 86400000, "days")
    print("capture:", s.envData["nbCapture"])
    np.savetxt("res/batterie.csv", [nd.battery.energyConsume for nd in s.envData["nodes"]], fmt="%4.4f")
    np.savetxt("res/colidSfPower.csv", s.envData["colidSfPower"], delimiter=",", fmt="%d")
    np.savetxt("res/timeOcc.csv", s.envData["timeOcc"] / (s.simTime / 100), delimiter=",", fmt="%f")
    head = "colide, capture,notHeard,total"
    np.savetxt("res/colid.csv",
               [[s.envData["collid"] - s.envData["nbCapture"] - s.envData["notHeard"], s.envData["nbCapture"],
                s.envData["notHeard"], s.envData["collid"]]], header=head, delimiter=",", fmt="%d,%d,%d,%d")

    head = "sf,power,energy,firstSentPacket,packetColid,packetTotalLost"
    for i in range(len(s.envData["nodes"])):
        np.savetxt("res/nodeLog/" + str(i) + ".csv", s.envData["log"][i], delimiter=",", fmt="%d,%d,%4.4f,%d,%d,%d",
                   header=head)
    drawGraphics(s)

    count = 0
    for nd in s.envData["nodes"]:
        count += nd.packetSent
    return count


stack = []
for it in range(1):
    res = main()
    stack.append(res)
print(sum(stack) / 1000)
