

class Battery:
    # capacity : capacité en milliAmpère-Heure de la batterie
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.energyConsume = 0

    # ajout de l'energie consomé à la baterie de la node
    # retur false si la totalité de la batterie est utilisée
    def useEnergy(self, energy: float):
        if self.capacity - self.energyConsume + energy > 0:
            self.energyConsume += energy
            return True
        else:
            return False
