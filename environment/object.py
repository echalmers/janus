from abc import ABC, abstractmethod
from random import random


class ObjectProperty(ABC):

    def movement_in_blocked(self):
        return False

    def movement_out_blocked(self):
        return False


class Solid(ObjectProperty):

    def movement_in_blocked(self):
        return True


class ImpedesMovement(ObjectProperty):

    def __init__(self, difficulty=0.5):
        self.difficulty = difficulty

    def movement_in_blocked(self):
        return random() < self.difficulty


class Sticky(ObjectProperty):

    def __init__(self, stickiness=0.5):
        self.stickiness = stickiness

    def movement_out_blocked(self):
        return random() < self.stickiness


class Hot(ObjectProperty):
    pass


class Edible(ObjectProperty):
    pass



class Object(ABC):

    def __init__(self, properties=None):
        if (properties is not None) and (not isinstance(properties, list)):
            raise Exception('properties must be passed in a list')
        if properties is None:
            properties = []
        self.properties = properties + [ObjectProperty()]

    def movement_in_blocked(self):
        return all([p.movement_in_blocked() for p in self.properties])

    def movement_out_blocked(self):
        return all([p.movement_out_blocked() for p in self.properties])

    def sense(self):
        return self.properties


class Wall(Object):

    def __init__(self):
        properties = [Solid()]
        super(Wall, self).__init__(properties)


class Trap(Object):

    def __init__(self):
        properties = [Sticky(stickiness=0.9)]
        super(Trap, self).__init__(properties)


class Lava(Object):

    def __init__(self):
        properties = [Hot(), Sticky(stickiness=0.25)]
        super(Lava, self).__init__(properties)


class Food(Object):
    def __init__(self):
        properties = [Edible(), Solid()]
        super(Food, self).__init__(properties)