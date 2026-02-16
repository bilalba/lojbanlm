# Lojban vs English: Reasoning Scaling Experiment

## Motivation

V2 results (small/medium models trained on literary text) showed:
- Lojban achieves 100% grammaticality at all sizes; English plateaus ~73-80%
- Val/test BPC consistently ~18-35% lower for Lojban
- At medium scale, Lojban samples showed emerging semantic coherence while
  English remained largely word salad

But "semantic coherence" is subjective and unmeasurable. We need **verifiable
reasoning tasks with binary right/wrong answers**.

**Core hypothesis**: Lojban's regular, unambiguous grammar allows small models
to develop verifiable reasoning capabilities at fewer parameters than English.
The scaling curve for reasoning accuracy is shifted left for Lojban.

## Approach: bAbI-Style Reasoning Tasks

All 20 tasks from the bAbI benchmark (Weston et al. 2015), implemented as
template generators producing parallel English/Lojban examples.

Properties:
- Exact correct answers (no subjective evaluation)
- Template-generated (no LLM needed, guaranteed correct)
- Both languages express identical logical content
- Difficulty scales across 20 task types
- Evaluation is exact-match accuracy

## The 20 Tasks

### Task 1: Single Supporting Fact

Lookup a single fact from context.

```
English:
  Mary went to the kitchen.
  John went to the garden.
  Where is Mary? kitchen

Lojban:
  la .maris. pu klama lo jukpa kumfa
  la .djan. pu klama lo purdi
  la .maris. cu zvati ma? lo jukpa kumfa
```

### Task 2: Two Supporting Facts

Two facts needed: who has object + where is that person.

```
English:
  John is in the playground.
  John picked up the football.
  Bob went to the kitchen.
  Where is the football? playground

Lojban:
  la .djan. cu zvati lo kelci stuzi
  la .djan. pu lebna lo bolci
  la .bab. pu klama lo jukpa kumfa
  lo bolci cu zvati ma? lo kelci stuzi
```

### Task 3: Three Supporting Facts

Three facts must be chained to answer.

```
English:
  John picked up the apple.
  John went to the office.
  John went to the kitchen.
  Where is the apple? kitchen

Lojban:
  la .djan. pu lebna lo plise
  la .djan. pu klama lo briju
  la .djan. pu klama lo jukpa kumfa
  lo plise cu zvati ma? lo jukpa kumfa
```

### Task 4: Two Argument Relations

Track two directional arguments of a relation.

```
English:
  The office is north of the bedroom.
  The bedroom is north of the bathroom.
  What is north of the bedroom? office
  What is the bedroom north of? bathroom

Lojban:
  lo briju cu berti lo sipna kumfa
  lo sipna kumfa cu berti lo lumku'a
  ma berti lo sipna kumfa? lo briju
  lo sipna kumfa cu berti ma? lo lumku'a
```

### Task 5: Three Argument Relations

Like Task 4 but requires chaining three relational facts.

```
English:
  The office is north of the bedroom.
  The bedroom is north of the bathroom.
  The kitchen is east of the office.
  What is north of the bathroom? bedroom
  What is east of the bedroom? kitchen (via office)

Lojban:
  lo briju cu berti lo sipna kumfa
  lo sipna kumfa cu berti lo lumku'a
  lo jukpa kumfa cu stuna lo briju
  ma berti lo lumku'a? lo sipna kumfa
```

### Task 6: Yes/No Questions

Simple fact verification.

```
English:
  John went to the playground.
  Mary went to the kitchen.
  Is John in the playground? yes
  Is John in the kitchen? no

Lojban:
  la .djan. pu klama lo kelci stuzi
  la .maris. pu klama lo jukpa kumfa
  xu la .djan. cu zvati lo kelci stuzi? go'i
  xu la .djan. cu zvati lo jukpa kumfa? na go'i
```

### Task 7: Counting

Track object acquisition/dropping, answer with a number.

```
English:
  Daniel picked up the football.
  Daniel dropped the football.
  Daniel picked up the milk.
  How many objects is Daniel holding? one

Lojban:
  la .daniyl. pu lebna lo bolci
  la .daniyl. pu falcru lo bolci
  la .daniyl. pu lebna lo ladru
  la .daniyl. cu bevri xokau? pa
```

### Task 8: Lists/Sets

Like counting but must enumerate the items.

```
English:
  Daniel picked up the football.
  Daniel picked up the milk.
  What is Daniel holding? football milk

Lojban:
  la .daniyl. pu lebna lo bolci
  la .daniyl. pu lebna lo ladru
  la .daniyl. cu bevri ma? lo bolci lo ladru
```

### Task 9: Simple Negation

Handle negation and "no longer" statements.

```
English:
  Sandra traveled to the office.
  Fred is no longer in the office.
  Is Fred in the office? no
  Is Sandra in the office? yes

Lojban:
  la .sandr. pu klama lo briju
  la .fred. ca na zvati lo briju
  xu la .fred. cu zvati lo briju? na go'i
  xu la .sandr. cu zvati lo briju? go'i
```

### Task 10: Indefinite Knowledge

Handle uncertainty ("either X or Y" → answer "maybe").

```
English:
  John is either in the classroom or the playground.
  Sandra is in the garden.
  Is John in the classroom? maybe
  Is Sandra in the garden? yes

Lojban:
  la .djan. cu zvati lo ckule ji lo kelci stuzi
  la .sandr. cu zvati lo purdi
  xu la .djan. cu zvati lo ckule? ju'o cu'i
  xu la .sandr. cu zvati lo purdi? go'i
```

### Task 11: Basic Coreference

Resolve pronoun to antecedent.

```
English:
  Daniel was in the kitchen.
  Then he traveled to the studio.
  Where is Daniel? studio

Lojban:
  la .daniyl. pu zvati lo jukpa kumfa
  ba bo ri pu klama lo larcu kumfa
  la .daniyl. cu zvati ma? lo larcu kumfa
```

### Task 12: Conjunction

Handle "X and Y did Z", then track individually.

```
English:
  Mary and Jeff went to the kitchen.
  Then Jeff went to the park.
  Where is Mary? kitchen
  Where is Jeff? park

Lojban:
  la .maris. .e la .djef. pu klama lo jukpa kumfa
  ba bo la .djef. pu klama lo panka
  la .maris. cu zvati ma? lo jukpa kumfa
  la .djef. cu zvati ma? lo panka
```

### Task 13: Compound Coreference

Handle "they" referring to multiple entities.

```
English:
  Daniel and Sandra went to the office.
  Then they went to the garden.
  Where is Daniel? garden

Lojban:
  la .daniyl. .e la .sandr. pu klama lo briju
  ba bo ry pu klama lo purdi
  la .daniyl. cu zvati ma? lo purdi
```

### Task 14: Time Reasoning

Track locations across time periods, answer about a past state.

```
English:
  In the afternoon Julie went to the park.
  Yesterday Julie was at school.
  Where was Julie before the park? school

Lojban:
  ca lo donri la .djulис. pu klama lo panka
  lo prulamdei la .djulis. pu zvati lo ckule
  la .djulis. pu zvati ma pu lo nu klama lo panka? lo ckule
```

### Task 15: Basic Deduction

Apply a universal rule to a specific instance.

```
English:
  Sheep are afraid of wolves.
  Cats are afraid of dogs.
  Mice are afraid of cats.
  Gertrude is a mouse.
  What is Gertrude afraid of? cats

Lojban:
  ro lanme cu terpa lo labno
  ro mlatu cu terpa lo gerku
  ro smacu cu terpa lo mlatu
  la .gertrud. cu smacu
  la .gertrud. cu terpa ma? lo mlatu
```

### Task 16: Basic Induction

Observe pattern from examples, generalize.

```
English:
  Lily is a swan. Lily is white.
  Bernhard is a swan.
  What color is Bernhard? white

Lojban:
  la .lilis. cu cipnrdjakni .i la .lilis. cu blabi
  la .bernart. cu cipnrdjakni
  la .bernart. cu skari ma? blabi
```

### Task 17: Positional Reasoning

Spatial relations (left/right/above/below) with transitivity.

```
English:
  The triangle is to the right of the blue square.
  The red square is above the triangle.
  Is the red square to the right of the blue square? yes

Lojban:
  lo ciblu'a cu pritu lo blanu kubli
  lo xunre kubli cu gapru lo ciblu'a
  xu lo xunre kubli cu pritu lo blanu kubli? go'i
```

### Task 18: Size Reasoning

Transitivity of containment / size relations.

```
English:
  The football fits in the suitcase.
  The suitcase fits in the cupboard.
  Will the football fit in the cupboard? yes
  Will the cupboard fit in the suitcase? no

Lojban:
  lo bolci cu se vasru lo dakli
  lo dakli cu se vasru lo kabrydai
  xu lo bolci cu se vasru lo kabrydai? go'i
  xu lo kabrydai cu se vasru lo dakli? na go'i
```

### Task 19: Path Finding

Navigate a described spatial graph.

```
English:
  The kitchen is north of the hallway.
  The bathroom is west of the bedroom.
  How do you go from the hallway to the kitchen? north

Lojban:
  lo jukpa kumfa cu berti lo vrogai
  lo lumku'a cu stici lo sipna kumfa
  lo vrogai mo'i ma lo jukpa kumfa? lo berti
```

### Task 20: Agent's Motivations

Infer why an agent went somewhere (because they got an object there).

```
English:
  John went to the kitchen.
  John got the apple.
  Why did John go to the kitchen? apple

Lojban:
  la .djan. pu klama lo jukpa kumfa
  la .djan. pu cpacu lo plise
  la .djan. mu'i ma pu klama lo jukpa kumfa? lo plise
```

## Why Lojban Should Have an Advantage

Each task exercises grammar features where Lojban is unambiguous and English is not:

| Task area | English problem | Lojban advantage |
|-----------|----------------|------------------|
| Questions (6,10) | Word order inversion ("is X...?") | `xu` particle, fixed position |
| Negation (9) | "not", "no longer", "isn't", "doesn't" | `na` in fixed position |
| Coreference (11,13) | "he/she/they" ambiguous | `ri`/`ra`/`vo'a` scoped |
| Conjunction (12) | "X and Y went" + later "he" | `.e` conjunction, clear scope |
| Time (14) | "before", "yesterday", tense agreement | `pu`/`ca`/`ba` explicit tense |
| Quantifiers (15) | "all X are Y" vs "X are Y" | `ro` = universal, explicit |
| Spatial (4,5,17,19) | Preposition overloading ("in", "on", "at") | Distinct gismu per relation |
| Motivation (20) | "why did X...?" word order | `mu'i ma` fixed structure |
| Counting (7) | Irregular numbers, agreement | Regular number system |
| Indefinite (10) | "either...or" / "maybe" | `ji` (or), `ju'o cu'i` (maybe) |

## Vocabulary Pools

Templates draw from these pools. Pool sizes determine how many unique examples
each task can generate.

### Names (~25)

| English | Lojban |
|---------|--------|
| Mary | la .maris. |
| John | la .djan. |
| Alice | la .alis. |
| Bob | la .bab. |
| Tom | la .tam. |
| Eve | la .iv. |
| Sam | la .sam. |
| Kate | la .keit. |
| Susan | la .suz. |
| David | la .deiv. |
| Rick | la .rik. |
| Peter | la .pet. |
| Lucy | la .lus. |
| Frank | la .frank. |
| Nina | la .ninas. |
| Bill | la .bil. |
| Ann | la .an. |
| Daniel | la .daniyl. |
| Fred | la .fred. |
| Sandra | la .sandr. |
| Julie | la .djulis. |
| Bernhard | la .bernart. |
| Lily | la .lilis. |
| Gertrude | la .gertrud. |
| Jeff | la .djef. |

### Locations (~15)

| English | Lojban |
|---------|--------|
| kitchen | lo jukpa kumfa |
| garden | lo purdi |
| bedroom | lo sipna kumfa |
| bathroom | lo lumku'a |
| office | lo briju |
| school | lo ckule |
| park | lo panka |
| hallway | lo vrogai |
| playground | lo kelci stuzi |
| studio | lo larcu kumfa |
| market | lo zarci |
| restaurant | lo gusta |
| library | lo ckusro |
| hospital | lo spita |
| cellar | lo kumfa cnita |

### Objects (~15)

| English | Lojban |
|---------|--------|
| ball | lo bolci |
| apple | lo plise |
| milk | lo ladru |
| book | lo cukta |
| key | lo ckiku |
| box | lo tanxe |
| bag | lo dakli |
| hat | lo mapku |
| cup | lo kabri |
| pen | lo penbi |
| shoe | lo cutci |
| bottle | lo botpi |
| knife | lo dakfu |
| coin | lo sicni |
| ring | lo djine |

### Animals (~12)

| English | Lojban |
|---------|--------|
| cat | mlatu |
| dog | gerku |
| bird | cipni |
| fish | finpe |
| mouse | smacu |
| horse | xirma |
| sheep | lanme |
| wolf | labno |
| lion | cinfo |
| bear | cribe |
| rabbit | ractu |
| snake | since |

### Colors (~6)

| English | Lojban |
|---------|--------|
| white | blabi |
| black | xekri |
| red | xunre |
| blue | blanu |
| green | crino |
| yellow | pelxu |

### Directions (~6)

| English | Lojban |
|---------|--------|
| north | berti |
| south | snanu |
| east | stuna |
| west | stici |
| above | gapru |
| below | cnita |

### Shapes (~4, for positional reasoning)

| English | Lojban |
|---------|--------|
| triangle | ciblu'a |
| square | kubli |
| circle | cukla |
| rectangle | kurfa |

## Unique Example Counts

With the pools above, approximate unique examples per task (accounting for
distractor combinations and ordering):

| Task | Core combinations | With distractors/ordering | Enough for 1K? |
|------|-------------------|---------------------------|-----------------|
| 1. Single fact | 25 × 15 = 375 | × distractor combos ≫ 1K | yes |
| 2. Two facts | 25 × 15 × 15 = 5,625 | easily | yes |
| 3. Three facts | 25 × 15 × 15 × 15 | easily | yes |
| 4. Two arg relations | 15 × 14 × 6 = 1,260 | yes | yes |
| 5. Three arg relations | 15 × 14 × 13 × 6 | easily | yes |
| 6. Yes/No | 25 × 15 = 375 | × distractor combos ≫ 1K | yes |
| 7. Counting | 25 × C(15,1..4) | easily | yes |
| 8. Lists | 25 × C(15,2..4) | easily | yes |
| 9. Negation | 25 × 15 × 14 | easily | yes |
| 10. Indefinite | 25 × 15 × 14 | easily | yes |
| 11. Coreference | 25 × 15 × 14 | easily | yes |
| 12. Conjunction | 25 × 24 × 15 × 14 | easily | yes |
| 13. Compound coref | 25 × 24 × 15 × 14 | easily | yes |
| 14. Time | 25 × 15 × 14 × 3 periods | easily | yes |
| 15. Deduction | 12 × 11 × ... rules × 25 | easily | yes |
| 16. Induction | 12 × 6 × 25 × 24 | easily | yes |
| 17. Positional | 4 × 6 × 6 combos | ~580 core, tight | marginal |
| 18. Size | 15 × 14 × 13 chains | easily | yes |
| 19. Path finding | 15^4 graph configs | easily | yes |
| 20. Motivation | 25 × 15 × 15 | easily | yes |

Task 17 (positional reasoning) is the tightest — may need to expand shape/color
vocabulary or accept ~500 training examples for that task.

## Data Generation

Python template scripts that generate parallel English + Lojban for each task.
No LLM involved. All Lojban validated with camxes.

**Per task**: 1,000 train / 200 val / 200 test (or as many unique examples as
the vocabulary supports, whichever is smaller).

**Format**: Each example is a single text sequence:

```
[fact 1]
[fact 2]
...
[question] [answer]
```

The model is trained on the full sequence. At eval time, we feed everything up
to the answer position and check if the model's greedy prediction matches.

### Training Regimes

Two options to explore:

**A. Per-task training**: One model per task. Tests whether the model can learn
each reasoning pattern in isolation. Cleaner signal but 20× the runs.

**B. Multi-task training**: One model trained on all tasks mixed together. Tests
whether the model can learn multiple reasoning patterns simultaneously. More
realistic and fewer runs, but harder tasks may be drowned out by easier ones.

Start with per-task (A) for clean scaling curves, then run multi-task (B) to
test generalization.

## Model Configurations

Character-level GPT, same architecture as V2:

| Size | d_model | n_layer | n_head | ~Params |
|------|---------|---------|--------|---------|
| Tiny | 64 | 2 | 2 | ~100K |
| XSmall | 96 | 3 | 3 | ~300K |
| Small | 128 | 4 | 2 | ~800K |
| Medium | 192 | 4 | 4 | ~1.8M |
| Large | 256 | 4 | 4 | ~3.2M |
| XLarge | 384 | 6 | 6 | ~10M |

Context length: 512 chars.
Seeds: 3 per configuration (42, 137, 2024).
Dropout: 0.10 across the board (large diverse training data, less overfitting risk).

### Per-task runs
6 sizes × 2 languages × 3 seeds × 20 tasks = 720 runs (but each is small/fast).

### Multi-task runs
6 sizes × 2 languages × 3 seeds = 36 runs.

## Evaluation

### Primary Metric: Exact-Match Accuracy

1. Feed model: story + question (everything up to the answer)
2. Greedy decode the answer (no temperature)
3. Compare to ground truth
4. Score: 1 if exact match, 0 otherwise

### Analysis

**Scaling curves**: For each task, plot accuracy (y) vs params (x, log scale),
with separate lines for English and Lojban.

**Parameter efficiency ratio**: For each task, find the smallest size where
accuracy > 90%. Report English_params / Lojban_params. If this ratio is
consistently > 1, the hypothesis is confirmed.

**Task difficulty ordering**: Rank tasks by difficulty (param count needed for
90%) for each language. Do they differ? If Lojban finds negation or quantifier
tasks disproportionately easier, that reveals which grammar features matter most.

**Statistical tests**: Paired t-tests across seeds at each size. Cohen's d for
effect sizes.

**Multi-task generalization**: Does training on all tasks help or hurt
individual task performance? Does the multi-task advantage differ by language?

## What Success Looks Like

**Strong**: Lojban reaches 90% accuracy at 2-4× fewer parameters across most
tasks. Clear leftward shift of the scaling curve.

**Moderate**: Lojban advantage on specific task families (negation, quantifiers,
spatial reasoning) but not others. Identifies which grammar features help.

**Null**: Similar scaling curves for both languages. Grammar regularity doesn't
affect reasoning emergence. (Still publishable — "grammar is free but doesn't
help reasoning" is a meaningful negative result.)

## Lojban Validation

All generated Lojban must parse with camxes. The template generators should
be designed so output is grammatically correct by construction, but camxes
validation serves as a safety net.

Some tasks need Lojban-specific attention:
- Task 10 (indefinite): `ji` usage and `ju'o cu'i` response need verification
- Task 11/13 (coreference): `ri`/`ra` scoping rules are subtle
- Task 14 (time): tense interaction with `pu`/`ca`/`ba` needs care
- Task 17 (positional): spatial gismu may need lujvo for some shapes

**These should be validated with a Lojban speaker or exhaustive camxes testing
before running experiments.**

## Separate Project: LLM Translation Pipeline

This experiment is entirely template-based and does NOT need LLM translation.

A separate LLM translation pipeline (English → Lojban via Claude/GPT-4 +
camxes validation) would enable future work:
- Naturalistic Lojban text at scale (V3 literary experiment)
- More complex reasoning tasks beyond what templates can express
- Domain-specific Lojban corpus creation

That pipeline needs its own development and validation:
- Translation accuracy testing (not just grammar — semantic fidelity)
- Camxes pass rate on raw LLM output
- Human evaluation on sample translations
- Cost and throughput benchmarking

It should be built and validated independently before being used for
training data.

## Dependencies

- Python 3, PyTorch
- Node.js (for camxes validation)
- No Java/LanguageTool (accuracy is the metric, not grammar checking)
- No LLM API access (template generation only)

## Implementation Order

1. Build and validate vocabulary pools (especially Lojban — confirm all
   gismu/lujvo are correct and parse)
2. Write template generators for tasks 1, 6, 9, 15 first (one from each
   major category: lookup, yes/no, negation, deduction)
3. Validate with camxes
4. Sanity check: train small model on task 1 — does it learn?
5. Implement remaining 16 task generators
6. Run per-task experiment (720 runs)
7. Run multi-task experiment (36 runs)
8. Analysis and visualization
9. Write up results
