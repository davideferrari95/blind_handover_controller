from typing import List

class Object:

    def __init__(self, name:str, over_pick_position:List[float], pick_position:List[float]):

        # Object Name, Pick Position, Place Position
        self.name: str = name
        self.over_pick_position: List[float] = over_pick_position
        self.pick_position: List[float] = pick_position

    def getName(self):
        return self.name

    def getOverPickPosition(self):
        return self.over_pick_position

    def getPickPosition(self):
        return self.pick_position

available_objects = [
    'screwdriver',
    'scissors',
    'pillars',
    'box'
]

object_list = [
    Object('screwdriver', [-4.12335759798158, -1.532313005333282, 2.228082005177633, -2.270500799218649, -1.565302197133199, -0.9527295271502894], [-4.129600111638204, -1.41463530183348, 2.287502352391378, -2.447676321069234, -1.5660713354693812, -0.9582765738116663]),
    Object('scissors',    [-4.28912586370577, -1.335634545688965, 1.881310764943258, -2.120077749291891, -1.564835850392476, -1.1186102072345179], [-4.315095965062277, -1.23264618337664, 1.993840042744771, -2.335579057733053, -1.5654948393451136, -1.1438449064837855]),
    Object('pillars',     [-3.68301994005312, -1.628258844415182, 2.325937096272604, -2.272941728631490, -1.566789452229635, -0.5125668684588831], [-3.683249775563375, -1.48723919809375, 2.397099320088522, -2.485088010827535, -1.5676410833941858, -0.5119169394122522]),
]

# Obtain Object Pick Positions
def get_object_pick_positions(object_name: str) -> List[float]:

    # Return Object Pick Position if Object found in Object List (by Name) else None
    return next(((obj.getOverPickPosition(), obj.getPickPosition()) for obj in object_list if obj.getName() == object_name), None)
