from typing import List

class Object:

    def __init__(self, name: str, pick_position: List[float]):

        # Object Name, Pick Position, Place Position
        self.name: str = name
        self.pick_position: List[float] = pick_position

    def getName(self):
        return self.name

    def getPickPosition(self):
        return self.pick_position

available_objects = [
    'screwdriver',
    'scissors',
    'pillars',
    'box'
]

object_list = [
    Object('screwdriver', [-3.692266289387838, -1.5014120799354096, 2.3944106737719935, -2.464505811730856, -1.5677226225482386, -0.4507320976257324]),
    Object('scissors',    [-3.692266289387838, -1.5014120799354096, 2.3944106737719935, -2.464505811730856, -1.5677226225482386, -0.4507320976257324]),
    Object('pillars',     [-3.692266289387838, -1.5014120799354096, 2.3944106737719935, -2.464505811730856, -1.5677226225482386, -0.4507320976257324]),
    Object('box',         [-3.692266289387838, -1.5014120799354096, 2.3944106737719935, -2.464505811730856, -1.5677226225482386, -0.4507320976257324]),
]

# Obtain Object Pick Position
def get_object_pick_position(object_name: str) -> List[float]:

    # Return Object Pick Position if Object found in Object List (by Name) else None
    return next((obj.getPickPosition() for obj in object_list if obj.getName() == object_name), None)
