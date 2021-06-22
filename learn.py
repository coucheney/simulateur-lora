import random

"""
La node ne change pas les paramètre de la Node
Sert également de classe de base pour la hiérachie des object qui permettent le choix des paramètres
"""
class Static:
    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int):
        return SF, power


# Si collision, les paramètre sont tirés aléatoirement dans les paramètres valides
class RandChoise(Static):
    def chooseParameter(self, power: int, SF: int, lostPacket: bool, validCombination: list, nbSend: int):
        if lostPacket:
            return random.choice(validCombination)
        else:
            return SF, power
