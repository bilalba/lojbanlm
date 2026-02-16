"""Vocabulary pools for bAbI task generation.

Each pool entry is a tuple of (english, lojban) or (english, lojban, metadata).
Pools are split into train and test (held-out) subsets. The split index marks
where train ends and test-only begins.
"""


# (english_name, lojban_name, pronoun)
ALL_NAMES = [
    ("Mary", "la .maris.", "she"),
    ("John", "la .djan.", "he"),
    ("Alice", "la .alis.", "she"),
    ("Bob", "la .bab.", "he"),
    ("Tom", "la .tam.", "he"),
    ("Eve", "la .iv.", "she"),
    ("Sam", "la .sam.", "he"),
    ("Kate", "la .keit.", "she"),
    ("Susan", "la .suz.", "she"),
    ("David", "la .deiv.", "he"),
    ("Rick", "la .rik.", "he"),
    ("Peter", "la .pet.", "he"),
    ("Lucy", "la .lus.", "she"),
    ("Frank", "la .frank.", "he"),
    ("Nina", "la .ninas.", "she"),
    ("Bill", "la .bil.", "he"),
    ("Ann", "la .an.", "she"),
    ("Daniel", "la .daniyl.", "he"),
    ("Fred", "la .fred.", "he"),
    ("Sandra", "la .sandr.", "she"),
    # --- held out for test ---
    ("Julie", "la .djulis.", "she"),
    ("Bernhard", "la .bernart.", "he"),
    ("Lily", "la .lilis.", "she"),
    ("Gertrude", "la .gertrud.", "she"),
    ("Jeff", "la .djef.", "he"),
]

# (english, lojban)
ALL_LOCATIONS = [
    ("kitchen", "lo jukpa kumfa"),
    ("garden", "lo purdi"),
    ("bedroom", "lo sipna kumfa"),
    ("bathroom", "lo lumku'a"),
    ("office", "lo briju"),
    ("school", "lo ckule"),
    ("park", "lo panka"),
    ("hallway", "lo vrogai"),
    ("playground", "lo kelci stuzi"),
    ("studio", "lo larcu kumfa"),
    ("market", "lo zarci"),
    ("restaurant", "lo gusta"),
    # --- held out ---
    ("library", "lo ckusro"),
    ("hospital", "lo spita"),
    ("cellar", "lo kumfa cnita"),
]

ALL_OBJECTS = [
    ("ball", "lo bolci"),
    ("apple", "lo plise"),
    ("milk", "lo ladru"),
    ("book", "lo cukta"),
    ("key", "lo ckiku"),
    ("box", "lo tanxe"),
    ("bag", "lo dakli"),
    ("hat", "lo mapku"),
    ("cup", "lo kabri"),
    ("pen", "lo penbi"),
    ("shoe", "lo cutci"),
    ("bottle", "lo botpi"),
    # --- held out ---
    ("knife", "lo dakfu"),
    ("coin", "lo sicni"),
    ("ring", "lo djine"),
]

ALL_ANIMALS = [
    ("cat", "mlatu", "cats"),
    ("dog", "gerku", "dogs"),
    ("bird", "cipni", "birds"),
    ("fish", "finpe", "fish"),
    ("mouse", "smacu", "mice"),
    ("horse", "xirma", "horses"),
    ("sheep", "lanme", "sheep"),
    ("wolf", "labno", "wolves"),
    ("lion", "cinfo", "lions"),
    # --- held out ---
    ("bear", "cribe", "bears"),
    ("rabbit", "ractu", "rabbits"),
    ("snake", "since", "snakes"),
]

ALL_COLORS = [
    ("white", "blabi"),
    ("black", "xekri"),
    ("red", "xunre"),
    ("blue", "blanu"),
    ("green", "crino"),
    # --- held out ---
    ("yellow", "pelxu"),
]

ALL_DIRECTIONS = [
    ("north", "berti"),
    ("south", "snanu"),
    ("east", "stuna"),
    ("west", "stici"),
    ("above", "gapru"),
    # --- held out ---
    ("below", "cnita"),
]

ALL_SHAPES = [
    ("triangle", "ciblu'a"),
    ("square", "kubli"),
    ("circle", "cukla"),
    # --- held out ---
    ("rectangle", "kurfa"),
]

# Spatial relations for positional reasoning (task 17)
# These are separate from compass directions
POSITIONAL_RELATIONS = [
    ("to the left of", "zunle"),
    ("to the right of", "pritu"),
    ("above", "gapru"),
    ("below", "cnita"),
]

# Numbers for counting (task 7)
NUMBERS_EN = ["none", "one", "two", "three", "four"]
NUMBERS_LJ = ["no", "pa", "re", "ci", "vo"]

# Time periods in chronological order (task 14)
TIME_PERIODS = [
    ("yesterday", "ca lo prulamdei"),
    ("this morning", "ca lo cerni"),
    ("in the afternoon", "ca lo donri"),
    ("this evening", "ca lo vanci"),
]

# ---------- Train / Test splits ----------

_NAME_SPLIT = 20
_LOC_SPLIT = 12
_OBJ_SPLIT = 12
_ANIMAL_SPLIT = 9
_COLOR_SPLIT = 5
_DIR_SPLIT = 5
_SHAPE_SPLIT = 3

TRAIN_NAMES = ALL_NAMES[:_NAME_SPLIT]
TEST_NAMES = ALL_NAMES[_NAME_SPLIT:]

TRAIN_LOCATIONS = ALL_LOCATIONS[:_LOC_SPLIT]
TEST_LOCATIONS = ALL_LOCATIONS[_LOC_SPLIT:]

TRAIN_OBJECTS = ALL_OBJECTS[:_OBJ_SPLIT]
TEST_OBJECTS = ALL_OBJECTS[_OBJ_SPLIT:]

TRAIN_ANIMALS = ALL_ANIMALS[:_ANIMAL_SPLIT]
TEST_ANIMALS = ALL_ANIMALS[_ANIMAL_SPLIT:]

TRAIN_COLORS = ALL_COLORS[:_COLOR_SPLIT]
TEST_COLORS = ALL_COLORS[_COLOR_SPLIT:]

TRAIN_DIRECTIONS = ALL_DIRECTIONS[:_DIR_SPLIT]
TEST_DIRECTIONS = ALL_DIRECTIONS[_DIR_SPLIT:]

TRAIN_SHAPES = ALL_SHAPES[:_SHAPE_SPLIT]
TEST_SHAPES = ALL_SHAPES[_SHAPE_SPLIT:]


class Vocab:
    """Container for a set of vocabulary pools."""

    def __init__(self, names, locations, objects, animals, colors,
                 directions, shapes):
        self.names = list(names)
        self.locations = list(locations)
        self.objects = list(objects)
        self.animals = list(animals)
        self.colors = list(colors)
        self.directions = list(directions)
        self.shapes = list(shapes)


TRAIN_VOCAB = Vocab(
    names=TRAIN_NAMES,
    locations=TRAIN_LOCATIONS,
    objects=TRAIN_OBJECTS,
    animals=TRAIN_ANIMALS,
    colors=TRAIN_COLORS,
    directions=TRAIN_DIRECTIONS,
    shapes=TRAIN_SHAPES,
)

FULL_VOCAB = Vocab(
    names=ALL_NAMES,
    locations=ALL_LOCATIONS,
    objects=ALL_OBJECTS,
    animals=ALL_ANIMALS,
    colors=ALL_COLORS,
    directions=ALL_DIRECTIONS,
    shapes=ALL_SHAPES,
)

# Set of all held-out English words (for filtering unseen test examples)
HELD_OUT_ENGLISH = set()
for pool_all, split in [
    (ALL_NAMES, _NAME_SPLIT),
    (ALL_LOCATIONS, _LOC_SPLIT),
    (ALL_OBJECTS, _OBJ_SPLIT),
    (ALL_ANIMALS, _ANIMAL_SPLIT),
    (ALL_COLORS, _COLOR_SPLIT),
    (ALL_DIRECTIONS, _DIR_SPLIT),
    (ALL_SHAPES, _SHAPE_SPLIT),
]:
    for item in pool_all[split:]:
        HELD_OUT_ENGLISH.add(item[0])
