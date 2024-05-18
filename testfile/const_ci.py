
class RangeDict(dict):
    def __getitem__(self, item):
        if not isinstance(item, range): # or xrange in Python 2
            for key in self:
                if item in key:
                    return self[key]
            raise KeyError(item)
        else:
            return super().__getitem__(item) # or super(RangeDict, self) for Python 2


CLASS_TYPE_INDEX = {
    "forward": 0,
    "back": 1,
    "stand": 2,
    "right": 3,
    "left": 4
}

INDEX_CLASS_TYPE = {
    0: "forward",
    1: "back",
    2: "stand",
    3: "right",
    4: "left"
}

EXCEPT_NAMES = [
    "stand",
    "right",
    "left"
]

ANGLE_CLASSES_NUM = 4
ANGLE_INTERVAL = int(360/ANGLE_CLASSES_NUM)
ANGLE_TO_CLASS = {}

temp = 0
for i in range(0, 360, ANGLE_INTERVAL):
    ANGLE_TO_CLASS[range(i, i+ANGLE_INTERVAL)] = temp
    temp += 1
ANGLE_TO_CLASS = RangeDict(ANGLE_TO_CLASS)

CLASS_NUM = len(CLASS_TYPE_INDEX.keys()) - len(EXCEPT_NAMES)
REGRESS_NUM = 2


DATA_PATHS = [
    "./data/cyh_fb/",
    "./data/jhc_fb/",
    "./data/lsh_fb/",
    "./data/pdh_fb/"
]

# DATA_PATHS = [
#     "./data/cyh1/",
#     "./data/cyh2/",
#     "./data/lsh/",
#     "./data/pdh/",
#     "./data/yws/",
#     "./data/bic/",
#     "./data/jhc/",
#     "./data/mjy/",
#     "./data/osm/",
#     "./data/ph/",
#     "./data/pth/"
# ]

'''
DATA_PATHS = [
    "./data/test/"
]
'''





