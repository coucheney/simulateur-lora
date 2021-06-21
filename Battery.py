

class Battery:
    def __init__(self, capacity):
        self.capacity = capacity
        self.energyConsume = 0

    def useEnergy(self, energy: float):
        if self.capacity - self.energyConsume + energy > 0:
            self.energyConsume += energy
            return True
        else:
            return False
