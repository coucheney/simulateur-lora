import os
import shutil
import sys

import numpy as np

from BS import BS
from Event import timmerEvent
from Packet import Point
from graphic import drawGraphics
from simu import Simu
from parse import parseScenario, readSensitivity, loadTX, loadNodeConfig, saveConfig


# création de l'objet simulation et ajout des paramètre
def initSimulation():
    env = Simu()
    env.addData([], "nodes")
    env.addData(readSensitivity(), "sensi")
    env.addData(loadTX(), "TX")
    env.addData(200, "radius")
    env.addData([], "nodePerSF")
    env.addData(0, "send")
    env.addData(0, "collid")
    env.addData(BS(0, Point(0, 0)), "BS")
    env.addData([], "reemit")
    env.addData(0, "nbCapture")
    env.addData(0, "notHeard")
    env.addData(False, "activateBaseLearning")
    return env

def writeResults(env: Simu, simTime):
    print("send :", env.envData["send"])
    print("collid :", env.envData["collid"])
    saveConfig(env)
    lowestBatterie = 0
    for nd in env.envData["nodes"]:
        if lowestBatterie < nd.battery.energyConsume:
            lowestBatterie = nd.battery.energyConsume
    print(lowestBatterie, "MiliAmpère-heure")
    print("time: ", simTime / 86400000, "days")
    print("capture:", env.envData["nbCapture"])
    np.savetxt("res/batterie.csv", [nd.battery.energyConsume for nd in env.envData["nodes"]], fmt="%4.4f")
    np.savetxt("res/timeOcc.csv", env.envData["timeOcc"] / (env.simTime / 100), delimiter=",", fmt="%f")
    head = "colide, capture,notHeard,total"
    np.savetxt("res/colid.csv",
               [[env.envData["collid"], env.envData["nbCapture"],
                 env.envData["notHeard"], env.envData["collid"] + env.envData["nbCapture"] + env.envData["notHeard"]]], header=head, delimiter=",", fmt="%d,%d,%d,%d")
    head = "time,day,sf,power,energy,firstSentPacket,packetColid,packetTotalLost"
    for i in range(len(env.envData["nodes"])):
        np.savetxt("res/nodeLog/" + str(i) + ".csv", env.envData["log"][i], delimiter=",",
                   fmt="%d,%d,%d,%d,%4.4f,%d,%d,%d",
                   header=head)
    np.savetxt("res/firstSend.csv", env.envData["firstSend"], fmt="%d")
    saveConfig(env)
    drawGraphics(env)

def main():
    env = initSimulation()

    # récupération du temps de simulation (par défaut le temps de simulation est de 100 jours)
    simTime = 86400000 * 100  # 1 jours
    if len(sys.argv) >= 2:
        try:
            simTime = int(sys.argv[1])
        except ValueError:
            print("l'argument doit être un entier")
            exit()

    env.addEvent(timmerEvent(0, env, simTime, 0))
    loadNodeConfig(env)

    # ajout des event liée aux scénario
    parseScenario(env)

    # execution de la simulation
    env.addData(np.zeros(len(env.envData["nodes"])), "firstSend")
    while env.simTime < simTime:
        env.nextEvent()

    # création des dossiers de résultats
    if os.path.exists("graphic"):
        shutil.rmtree("graphic")
    os.makedirs("graphic")
    if os.path.exists("res"):
        shutil.rmtree("res")
    os.makedirs("res")
    os.makedirs("res/nodeLog")

    # écriture des résultats
    writeResults(env, simTime)

main()
