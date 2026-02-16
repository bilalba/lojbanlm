"""bAbI task generators producing parallel English/Lojban examples.

Each task function takes (rng, vocab, n) and returns a list of
(english_text, lojban_text) tuples. Examples are deduplicated.
"""

import random
from .vocab import (
    NUMBERS_EN, NUMBERS_LJ, TIME_PERIODS, POSITIONAL_RELATIONS,
)


# ---------- helpers ----------

def _sample(rng, pool, n):
    return rng.sample(pool, min(n, len(pool)))


def _dedup(examples, n):
    seen = set()
    out = []
    for en, lj in examples:
        if en not in seen:
            seen.add(en)
            out.append((en, lj))
            if len(out) >= n:
                break
    return out


def _shuffle_paired(rng, list_a, list_b):
    """Shuffle two lists in parallel."""
    combined = list(zip(list_a, list_b))
    rng.shuffle(combined)
    return [x[0] for x in combined], [x[1] for x in combined]


# Directions that take "of" in English spatial statements
# "north of X" but "above X" (not "above of X")
_NO_OF_DIRECTIONS = {"above", "below"}


def _dir_of(direction_en):
    """Return 'north of' or 'above' (no 'of' for above/below)."""
    if direction_en in _NO_OF_DIRECTIONS:
        return direction_en
    return f"{direction_en} of"


# ============================================================
# Task 1: Single Supporting Fact
# ============================================================

def task01(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        target = rng.choice(vocab.names)
        target_loc = rng.choice(vocab.locations)

        n_dist = rng.randint(1, 4)
        others = [p for p in vocab.names if p != target]
        distractors = [(d, rng.choice(vocab.locations))
                       for d in _sample(rng, others, n_dist)]

        facts = [(target, target_loc)] + distractors
        en_f = [f"{p[0]} went to the {l[0]}." for p, l in facts]
        lj_f = [f"{p[1]} pu klama {l[1]}" for p, l in facts]
        en_f, lj_f = _shuffle_paired(rng, en_f, lj_f)

        en = "\n".join(en_f + [f"Where is {target[0]}? {target_loc[0]}"])
        lj = "\n".join(lj_f + [f"{target[1]} cu zvati ma? {target_loc[1]}"])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 2: Two Supporting Facts
# ============================================================

def task02(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        loc = rng.choice(vocab.locations)
        obj = rng.choice(vocab.objects)

        n_dist = rng.randint(1, 3)
        others = [p for p in vocab.names if p != person]
        dist = [(d, rng.choice(vocab.locations))
                for d in _sample(rng, others, n_dist)]

        en_f = [f"{person[0]} is in the {loc[0]}.",
                f"{person[0]} picked up the {obj[0]}."]
        lj_f = [f"{person[1]} cu zvati {loc[1]}",
                f"{person[1]} pu lebna {obj[1]}"]
        for d, dl in dist:
            en_f.append(f"{d[0]} went to the {dl[0]}.")
            lj_f.append(f"{d[1]} pu klama {dl[1]}")

        en = "\n".join(en_f + [f"Where is the {obj[0]}? {loc[0]}"])
        lj = "\n".join(lj_f + [f"{obj[1]} cu zvati ma? {loc[1]}"])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 3: Three Supporting Facts
# ============================================================

def task03(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        obj = rng.choice(vocab.objects)
        locs = _sample(rng, vocab.locations, 2)

        n_dist = rng.randint(0, 2)
        others = [p for p in vocab.names if p != person]
        dist = [(d, rng.choice(vocab.locations))
                for d in _sample(rng, others, n_dist)]

        # person picks up obj, goes to loc0, then loc1 → obj at loc1
        en_f = [f"{person[0]} picked up the {obj[0]}.",
                f"{person[0]} went to the {locs[0][0]}.",
                f"{person[0]} went to the {locs[1][0]}."]
        lj_f = [f"{person[1]} pu lebna {obj[1]}",
                f"{person[1]} pu klama {locs[0][1]}",
                f"{person[1]} pu klama {locs[1][1]}"]
        for d, dl in dist:
            pos = rng.randint(0, len(en_f))
            en_f.insert(pos, f"{d[0]} went to the {dl[0]}.")
            lj_f.insert(pos, f"{d[1]} pu klama {dl[1]}")

        en = "\n".join(en_f + [f"Where is the {obj[0]}? {locs[1][0]}"])
        lj = "\n".join(lj_f + [f"{obj[1]} cu zvati ma? {locs[1][1]}"])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 4: Two Argument Relations
# ============================================================

def task04(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        locs = _sample(rng, vocab.locations, 3)
        d = rng.choice(vocab.directions)

        # L0 is [dir] of L1 ; L1 is [dir] of L2
        dof = _dir_of(d[0])
        en_f = [f"The {locs[0][0]} is {dof} the {locs[1][0]}.",
                f"The {locs[1][0]} is {dof} the {locs[2][0]}."]
        lj_f = [f"{locs[0][1]} cu {d[1]} {locs[1][1]}",
                f"{locs[1][1]} cu {d[1]} {locs[2][1]}"]

        if rng.random() < 0.5:
            en_q = f"What is {dof} the {locs[1][0]}? {locs[0][0]}"
            lj_q = f"ma {d[1]} {locs[1][1]}? {locs[0][1]}"
        else:
            en_q = f"What is the {locs[1][0]} {dof}? {locs[2][0]}"
            lj_q = f"{locs[1][1]} cu {d[1]} ma? {locs[2][1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 5: Three Argument Relations
# ============================================================

def task05(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        locs = _sample(rng, vocab.locations, 4)
        dirs = _sample(rng, vocab.directions, 2)

        dof0 = _dir_of(dirs[0][0])
        dof1 = _dir_of(dirs[1][0])
        en_f = [f"The {locs[0][0]} is {dof0} the {locs[1][0]}.",
                f"The {locs[1][0]} is {dof0} the {locs[2][0]}.",
                f"The {locs[3][0]} is {dof1} the {locs[0][0]}."]
        lj_f = [f"{locs[0][1]} cu {dirs[0][1]} {locs[1][1]}",
                f"{locs[1][1]} cu {dirs[0][1]} {locs[2][1]}",
                f"{locs[3][1]} cu {dirs[1][1]} {locs[0][1]}"]

        en_f, lj_f = _shuffle_paired(rng, en_f, lj_f)

        # 1-hop lookup from one of the 3 facts
        q_type = rng.randint(0, 2)
        if q_type == 0:
            en_q = f"What is {dof0} the {locs[1][0]}? {locs[0][0]}"
            lj_q = f"ma {dirs[0][1]} {locs[1][1]}? {locs[0][1]}"
        elif q_type == 1:
            en_q = f"What is {dof0} the {locs[2][0]}? {locs[1][0]}"
            lj_q = f"ma {dirs[0][1]} {locs[2][1]}? {locs[1][1]}"
        else:
            en_q = f"What is {dof1} the {locs[0][0]}? {locs[3][0]}"
            lj_q = f"ma {dirs[1][1]} {locs[0][1]}? {locs[3][1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 6: Yes/No Questions
# ============================================================

def task06(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        n_people = rng.randint(2, 4)
        people = _sample(rng, vocab.names, n_people)
        person_locs = [(p, rng.choice(vocab.locations)) for p in people]

        en_f = [f"{p[0]} went to the {l[0]}." for p, l in person_locs]
        lj_f = [f"{p[1]} pu klama {l[1]}" for p, l in person_locs]

        target, target_loc = rng.choice(person_locs)

        if rng.random() < 0.5:
            en_q = f"Is {target[0]} in the {target_loc[0]}? yes"
            lj_q = f"xu {target[1]} cu zvati {target_loc[1]}? go'i"
        else:
            wrong = rng.choice([l for l in vocab.locations if l != target_loc])
            en_q = f"Is {target[0]} in the {wrong[0]}? no"
            lj_q = f"xu {target[1]} cu zvati {wrong[1]}? na go'i"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 7: Counting
# ============================================================

def task07(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        available = list(vocab.objects)
        rng.shuffle(available)

        held = []
        en_f, lj_f = [], []
        n_actions = rng.randint(2, 6)

        for _ in range(n_actions):
            if held and rng.random() < 0.35:
                obj = rng.choice(held)
                held.remove(obj)
                en_f.append(f"{person[0]} dropped the {obj[0]}.")
                lj_f.append(f"{person[1]} pu falcru {obj[1]}")
            else:
                cands = [o for o in available if o not in held]
                if not cands:
                    continue
                obj = rng.choice(cands)
                held.append(obj)
                en_f.append(f"{person[0]} picked up the {obj[0]}.")
                lj_f.append(f"{person[1]} pu lebna {obj[1]}")

        count = len(held)
        if count > 4:
            continue

        en_q = f"How many objects is {person[0]} holding? {NUMBERS_EN[count]}"
        lj_q = f"{person[1]} cu bevri xokau? {NUMBERS_LJ[count]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 8: Lists / Sets
# ============================================================

def task08(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        n_obj = rng.randint(1, 4)
        objs = _sample(rng, vocab.objects, n_obj)

        en_f = [f"{person[0]} picked up the {o[0]}." for o in objs]
        lj_f = [f"{person[1]} pu lebna {o[1]}" for o in objs]

        # optionally add a pick-up-then-drop distractor
        if rng.random() < 0.3:
            extras = [o for o in vocab.objects if o not in objs]
            if extras:
                extra = rng.choice(extras)
                en_f.append(f"{person[0]} picked up the {extra[0]}.")
                en_f.append(f"{person[0]} dropped the {extra[0]}.")
                lj_f.append(f"{person[1]} pu lebna {extra[1]}")
                lj_f.append(f"{person[1]} pu falcru {extra[1]}")

        en_ans = " ".join(o[0] for o in objs)
        lj_ans = " ".join(o[1] for o in objs)

        en_q = f"What is {person[0]} holding? {en_ans}"
        lj_q = f"{person[1]} cu bevri ma? {lj_ans}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 9: Simple Negation
# ============================================================

def task09(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        people = _sample(rng, vocab.names, rng.randint(2, 4))
        loc = rng.choice(vocab.locations)

        person_a = people[0]   # goes there
        person_b = people[1]   # no longer there

        en_f = [f"{person_a[0]} traveled to the {loc[0]}.",
                f"{person_b[0]} is no longer in the {loc[0]}."]
        lj_f = [f"{person_a[1]} pu klama {loc[1]}",
                f"{person_b[1]} ca na zvati {loc[1]}"]

        for p in people[2:]:
            dl = rng.choice(vocab.locations)
            en_f.append(f"{p[0]} went to the {dl[0]}.")
            lj_f.append(f"{p[1]} pu klama {dl[1]}")

        if rng.random() < 0.5:
            en_q = f"Is {person_b[0]} in the {loc[0]}? no"
            lj_q = f"xu {person_b[1]} cu zvati {loc[1]}? na go'i"
        else:
            en_q = f"Is {person_a[0]} in the {loc[0]}? yes"
            lj_q = f"xu {person_a[1]} cu zvati {loc[1]}? go'i"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 10: Indefinite Knowledge
# ============================================================

def task10(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        people = _sample(rng, vocab.names, rng.randint(2, 3))
        indef_locs = _sample(rng, vocab.locations, 2)
        def_loc = rng.choice(vocab.locations)

        en_f = [
            f"{people[0][0]} is either in the {indef_locs[0][0]} or the {indef_locs[1][0]}.",
            f"{people[1][0]} is in the {def_loc[0]}.",
        ]
        lj_f = [
            f"{people[0][1]} cu zvati {indef_locs[0][1]} ji {indef_locs[1][1]}",
            f"{people[1][1]} cu zvati {def_loc[1]}",
        ]

        r = rng.random()
        if r < 0.33:
            ask_loc = rng.choice(indef_locs)
            en_q = f"Is {people[0][0]} in the {ask_loc[0]}? maybe"
            lj_q = f"xu {people[0][1]} cu zvati {ask_loc[1]}? ju'o cu'i"
        elif r < 0.66:
            en_q = f"Is {people[1][0]} in the {def_loc[0]}? yes"
            lj_q = f"xu {people[1][1]} cu zvati {def_loc[1]}? go'i"
        else:
            wrong = rng.choice([l for l in vocab.locations if l != def_loc])
            en_q = f"Is {people[1][0]} in the {wrong[0]}? no"
            lj_q = f"xu {people[1][1]} cu zvati {wrong[1]}? na go'i"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 11: Basic Coreference
# ============================================================

def task11(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        locs = _sample(rng, vocab.locations, 2)
        pronoun = person[2]  # "he" or "she"

        # Distractors go BEFORE the coreference pair so that
        # English pronoun is unambiguous (only recent antecedent)
        # and Lojban ri refers to the correct sumti.
        en_f, lj_f = [], []
        if rng.random() < 0.5:
            # pick distractor of opposite gender to keep English unambiguous
            opp = "she" if pronoun == "he" else "he"
            cands = [p for p in vocab.names if p != person and p[2] == opp]
            if cands:
                other = rng.choice(cands)
                dl = rng.choice(vocab.locations)
                en_f.append(f"{other[0]} went to the {dl[0]}.")
                lj_f.append(f"{other[1]} pu klama {dl[1]}")

        en_f += [f"{person[0]} was in the {locs[0][0]}.",
                 f"Then {pronoun} traveled to the {locs[1][0]}."]
        lj_f += [f"{person[1]} pu zvati {locs[0][1]}",
                 f"ba bo ri pu klama {locs[1][1]}"]

        en = "\n".join(en_f + [f"Where is {person[0]}? {locs[1][0]}"])
        lj = "\n".join(lj_f + [f"{person[1]} cu zvati ma? {locs[1][1]}"])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 12: Conjunction
# ============================================================

def task12(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        people = _sample(rng, vocab.names, 2)
        locs = _sample(rng, vocab.locations, 2)

        # Both go to loc0, then person[1] goes to loc1
        en_f = [
            f"{people[0][0]} and {people[1][0]} went to the {locs[0][0]}.",
            f"Then {people[1][0]} went to the {locs[1][0]}.",
        ]
        lj_f = [
            f"{people[0][1]} .e {people[1][1]} pu klama {locs[0][1]}",
            f"ba bo {people[1][1]} pu klama {locs[1][1]}",
        ]

        if rng.random() < 0.5:
            # ask about person who stayed
            en_q = f"Where is {people[0][0]}? {locs[0][0]}"
            lj_q = f"{people[0][1]} cu zvati ma? {locs[0][1]}"
        else:
            # ask about person who moved
            en_q = f"Where is {people[1][0]}? {locs[1][0]}"
            lj_q = f"{people[1][1]} cu zvati ma? {locs[1][1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 13: Compound Coreference
# ============================================================

def task13(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        people = _sample(rng, vocab.names, 2)
        locs = _sample(rng, vocab.locations, 2)

        # Both go to loc0, then "they" go to loc1
        en_f = [
            f"{people[0][0]} and {people[1][0]} went to the {locs[0][0]}.",
            f"Then they went to the {locs[1][0]}.",
        ]
        # In Lojban, ry refers back to the compound subject
        lj_f = [
            f"{people[0][1]} .e {people[1][1]} pu klama {locs[0][1]}",
            f"ba bo ry pu klama {locs[1][1]}",
        ]

        target = rng.choice(people)
        en_q = f"Where is {target[0]}? {locs[1][0]}"
        lj_q = f"{target[1]} cu zvati ma? {locs[1][1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 14: Time Reasoning
# ============================================================

def task14(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        # pick 2-3 time periods and assign locations
        n_periods = rng.randint(2, min(3, len(TIME_PERIODS)))
        period_idxs = sorted(rng.sample(range(len(TIME_PERIODS)), n_periods))
        time_locs = [(TIME_PERIODS[i], rng.choice(vocab.locations))
                     for i in period_idxs]

        en_f, lj_f = [], []
        for (t_en, t_lj), loc in time_locs:
            en_f.append(f"{t_en.capitalize()} {person[0]} went to the {loc[0]}.")
            lj_f.append(f"{t_lj} {person[1]} pu klama {loc[1]}")

        # Shuffle facts (out of chronological order adds difficulty)
        en_f, lj_f = _shuffle_paired(rng, en_f, lj_f)

        # Question: where was person before the LAST location?
        # Answer: the second-to-last location in chronological order
        later_loc = time_locs[-1][1]
        earlier_loc = time_locs[-2][1]

        en_q = f"Where was {person[0]} before the {later_loc[0]}? {earlier_loc[0]}"
        lj_q = (f"{person[1]} pu zvati ma pu lo nu klama {later_loc[1]}?"
                f" {earlier_loc[1]}")

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 15: Basic Deduction
# ============================================================

def task15(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        if len(vocab.animals) < 3:
            break
        # pick 3-5 animals and create fear chain: a0 fears a1, a1 fears a2, ...
        n_animals = rng.randint(3, min(5, len(vocab.animals)))
        animals = _sample(rng, vocab.animals, n_animals)

        # Rules: animals[i] fears animals[i+1]
        # Animal tuples: (singular, lojban, plural)
        en_f, lj_f = [], []
        for i in range(len(animals) - 1):
            prey, pred = animals[i], animals[i + 1]
            en_f.append(f"{prey[2].capitalize()} are afraid of {pred[2]}.")
            lj_f.append(f"ro {prey[1]} cu terpa lo {pred[1]}")

        # Pick a person and assign them one of the prey animals
        person = rng.choice(vocab.names)
        prey_idx = rng.randint(0, len(animals) - 2)
        animal_type = animals[prey_idx]
        feared = animals[prey_idx + 1]

        en_f.append(f"{person[0]} is a {animal_type[0]}.")
        lj_f.append(f"{person[1]} cu {animal_type[1]}")

        en_f, lj_f = _shuffle_paired(rng, en_f, lj_f)

        en_q = f"What is {person[0]} afraid of? {feared[2]}"
        lj_q = f"{person[1]} cu terpa ma? lo {feared[1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 16: Basic Induction
# ============================================================

def task16(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        animal = rng.choice(vocab.animals)
        color = rng.choice(vocab.colors)

        # 2-3 exemplars showing: X is a [type]. X is [color].
        n_ex = rng.randint(2, 3)
        names = _sample(rng, vocab.names, n_ex + 1)
        exemplars = names[:n_ex]
        target = names[n_ex]

        en_f, lj_f = [], []
        for nm in exemplars:
            en_f.append(f"{nm[0]} is a {animal[0]}. {nm[0]} is {color[0]}.")
            lj_f.append(
                f"{nm[1]} cu {animal[1]} .i {nm[1]} cu {color[1]}")

        en_f.append(f"{target[0]} is a {animal[0]}.")
        lj_f.append(f"{target[1]} cu {animal[1]}")

        en_q = f"What color is {target[0]}? {color[0]}"
        lj_q = f"{target[1]} cu skari ma? {color[1]}"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 17: Positional Reasoning
# ============================================================

def _make_colored_shape(color, shape):
    en = f"the {color[0]} {shape[0]}"
    lj = f"lo {color[1]} {shape[1]}"
    return (en, lj)


def task17(rng, vocab, n):
    examples = []
    for _ in range(n * 15):
        # Create 3 distinct colored shapes
        if len(vocab.colors) < 3 or len(vocab.shapes) < 1:
            break
        colors = _sample(rng, vocab.colors, 3)
        shapes = [rng.choice(vocab.shapes) for _ in range(3)]
        items = [_make_colored_shape(c, s) for c, s in zip(colors, shapes)]

        # Place on a line (single axis for clean transitivity)
        # Pick axis: left-right (0) or up-down (1)
        axis = rng.randint(0, 1)
        if axis == 0:
            rel = POSITIONAL_RELATIONS[1]   # "to the right of" / pritu
            inv = POSITIONAL_RELATIONS[0]   # "to the left of" / zunle
        else:
            rel = POSITIONAL_RELATIONS[2]   # "above" / gapru
            inv = POSITIONAL_RELATIONS[3]   # "below" / cnita

        # items[0] rel items[1], items[1] rel items[2]
        # So items[0] rel items[2] by transitivity
        en_f = [
            f"{items[0][0].capitalize()} is {rel[0]} {items[1][0]}.",
            f"{items[1][0].capitalize()} is {rel[0]} {items[2][0]}.",
        ]
        lj_f = [
            f"{items[0][1]} cu {rel[1]} {items[1][1]}",
            f"{items[1][1]} cu {rel[1]} {items[2][1]}",
        ]

        r = rng.random()
        if r < 0.33:
            # Transitive: items[0] rel items[2]? yes
            en_q = f"Is {items[0][0]} {rel[0]} {items[2][0]}? yes"
            lj_q = f"xu {items[0][1]} cu {rel[1]} {items[2][1]}? go'i"
        elif r < 0.66:
            # Inverse: items[2] rel items[0]? no
            en_q = f"Is {items[2][0]} {rel[0]} {items[0][0]}? no"
            lj_q = f"xu {items[2][1]} cu {rel[1]} {items[0][1]}? na go'i"
        else:
            # Direct lookup: items[0] rel items[1]? yes
            en_q = f"Is {items[0][0]} {rel[0]} {items[1][0]}? yes"
            lj_q = f"xu {items[0][1]} cu {rel[1]} {items[1][1]}? go'i"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 18: Size Reasoning
# ============================================================

def task18(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        objs = _sample(rng, vocab.objects, 3)
        # objs[0] fits in objs[1], objs[1] fits in objs[2]
        # → objs[0] fits in objs[2] (yes)
        # → objs[2] fits in objs[1] (no)

        en_f = [
            f"The {objs[0][0]} fits in the {objs[1][0]}.",
            f"The {objs[1][0]} fits in the {objs[2][0]}.",
        ]
        lj_f = [
            f"{objs[0][1]} cu se vasru {objs[1][1]}",
            f"{objs[1][1]} cu se vasru {objs[2][1]}",
        ]

        r = rng.random()
        if r < 0.33:
            # Transitive: small fits in large? yes
            en_q = f"Will the {objs[0][0]} fit in the {objs[2][0]}? yes"
            lj_q = f"xu {objs[0][1]} cu se vasru {objs[2][1]}? go'i"
        elif r < 0.66:
            # Inverse: large fits in small? no
            en_q = f"Will the {objs[2][0]} fit in the {objs[0][0]}? no"
            lj_q = f"xu {objs[2][1]} cu se vasru {objs[0][1]}? na go'i"
        else:
            # Direct: medium fits in large? yes
            en_q = f"Will the {objs[1][0]} fit in the {objs[2][0]}? yes"
            lj_q = f"xu {objs[1][1]} cu se vasru {objs[2][1]}? go'i"

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 19: Path Finding
# ============================================================

def task19(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        # Build small graph: 3-4 locations connected by directions
        n_edges = rng.randint(2, 3)
        locs = _sample(rng, vocab.locations, n_edges + 1)

        edges_en, edges_lj = [], []
        edge_list = []  # (from_loc, to_loc, direction)
        for i in range(n_edges):
            d = rng.choice(vocab.directions)
            src, dst = locs[i + 1], locs[i]
            # "dst is [dir] of src" means going from src in [dir] reaches dst
            dof = _dir_of(d[0])
            edges_en.append(
                f"The {dst[0]} is {dof} the {src[0]}.")
            edges_lj.append(
                f"{dst[1]} cu {d[1]} {src[1]}")
            edge_list.append((src, dst, d))

        edges_en, edges_lj = _shuffle_paired(rng, edges_en, edges_lj)

        # Question: how to go from src to dst? answer = direction
        src, dst, d = rng.choice(edge_list)
        en_q = (f"How do you go from the {src[0]} to the {dst[0]}?"
                f" {d[0]}")
        lj_q = f"{src[1]} mo'i ma {dst[1]}? lo {d[1]}"

        en = "\n".join(edges_en + [en_q])
        lj = "\n".join(edges_lj + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Task 20: Agent's Motivations
# ============================================================

def task20(rng, vocab, n):
    examples = []
    for _ in range(n * 10):
        person = rng.choice(vocab.names)
        loc = rng.choice(vocab.locations)
        obj = rng.choice(vocab.objects)

        # Person goes to location, gets object there
        en_f = [f"{person[0]} went to the {loc[0]}.",
                f"{person[0]} got the {obj[0]}."]
        lj_f = [f"{person[1]} pu klama {loc[1]}",
                f"{person[1]} pu cpacu {obj[1]}"]

        # Distractors: other people going places and getting things
        n_dist = rng.randint(0, 2)
        others = [p for p in vocab.names if p != person]
        for d in _sample(rng, others, n_dist):
            dl = rng.choice(vocab.locations)
            do = rng.choice(vocab.objects)
            en_f.append(f"{d[0]} went to the {dl[0]}.")
            en_f.append(f"{d[0]} got the {do[0]}.")
            lj_f.append(f"{d[1]} pu klama {dl[1]}")
            lj_f.append(f"{d[1]} pu cpacu {do[1]}")

        en_q = (f"Why did {person[0]} go to the {loc[0]}?"
                f" {obj[0]}")
        lj_q = (f"{person[1]} mu'i ma pu klama {loc[1]}?"
                f" {obj[1]}")

        en = "\n".join(en_f + [en_q])
        lj = "\n".join(lj_f + [lj_q])
        examples.append((en, lj))

    return _dedup(examples, n)


# ============================================================
# Registry
# ============================================================

TASK_GENERATORS = {
    1: ("single_supporting_fact", task01),
    2: ("two_supporting_facts", task02),
    3: ("three_supporting_facts", task03),
    4: ("two_argument_relations", task04),
    5: ("three_argument_relations", task05),
    6: ("yes_no_questions", task06),
    7: ("counting", task07),
    8: ("lists_sets", task08),
    9: ("simple_negation", task09),
    10: ("indefinite_knowledge", task10),
    11: ("basic_coreference", task11),
    12: ("conjunction", task12),
    13: ("compound_coreference", task13),
    14: ("time_reasoning", task14),
    15: ("basic_deduction", task15),
    16: ("basic_induction", task16),
    17: ("positional_reasoning", task17),
    18: ("size_reasoning", task18),
    19: ("path_finding", task19),
    20: ("agents_motivations", task20),
}
