

class Battery:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.energyConsume = 0

    # ajout de l'energie consomé à la baterie de la node
    def useEnergy(self, energy: float):
        if self.capacity - self.energyConsume + energy > 0:
            self.energyConsume += energy
            return True
        else:
            return False
