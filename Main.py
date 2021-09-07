import os
import shutil
import numpy as np

from BS import BS
from Event import timmerEvent
from Packet import Point
from graphic import drawGraphics
from simu import Simu
from parse import parseScenario, readSensitivity, loadTX, loadNodeConfig, saveConfig


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

def writeResults(s: Simu, simTime):
    print("send :", s.envData["send"])
    print("collid :", s.envData["collid"])
    saveConfig(s)
    lowestBatterie = 0
    for nd in s.envData["nodes"]:
        if lowestBatterie < nd.battery.energyConsume:
            lowestBatterie = nd.battery.energyConsume
    print(lowestBatterie, "MiliAmpère-heure")
    print("time: ", simTime / 86400000, "days")
    print("capture:", s.envData["nbCapture"])
    np.savetxt("res/batterie.csv", [nd.battery.energyConsume for nd in s.envData["nodes"]], fmt="%4.4f")
    np.savetxt("res/timeOcc.csv", s.envData["timeOcc"] / (s.simTime / 100), delimiter=",", fmt="%f")
    head = "colide, capture,notHeard,total"
    np.savetxt("res/colid.csv",
               [[s.envData["collid"], s.envData["nbCapture"],
                 s.envData["notHeard"], s.envData["collid"] + s.envData["nbCapture"] + s.envData["notHeard"]]], header=head, delimiter=",", fmt="%d,%d,%d,%d")
    head = "time,day,sf,power,energy,firstSentPacket,packetColid,packetTotalLost"
    for i in range(len(s.envData["nodes"])):
        np.savetxt("res/nodeLog/" + str(i) + ".csv", s.envData["log"][i], delimiter=",",
                   fmt="%d,%d,%d,%d,%4.4f,%d,%d,%d",
                   header=head)
    np.savetxt("res/firstSend.csv", s.envData["firstSend"], fmt="%d")

    drawGraphics(s)

def main():
    s = initSimulation()
    # simTime = 1800000000   # temp de l'article
    simTime = 86400000 * 10 # 1 jours
    s.addEvent(timmerEvent(0, s, simTime, 0))
    loadNodeConfig(s)

    # ajout des event liée aux scénario
    parseScenario(s)

    # execution de la simulation
    s.addData(np.zeros(len(s.envData["nodes"])), "firstSend")
    while s.simTime < simTime:
        s.nextEvent()

    # création des dossiers de résultats
    if os.path.exists("graphic"):
        shutil.rmtree("graphic")
    os.makedirs("graphic")
    if os.path.exists("res"):
        shutil.rmtree("res")
    os.makedirs("res")
    os.makedirs("res/nodeLog")

    # écriture des résultats
    writeResults(s, simTime)

main()
