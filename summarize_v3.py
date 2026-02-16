import json, os

base = os.path.expanduser("~/lojban_experiment/results/v3")
header = "{:<6} {:<8} {:<5} {:>7} {:>7} {:>6} {:>7} {:>7}".format(
    "Size", "Lang", "Seed", "ValBPC", "TstBPC", "Gram%", "bAbI_S", "bAbI_U")
print(header)
print("-" * len(header))
for size in ["nano", "micro", "mini", "small", "base"]:
    for lang in ["english", "lojban"]:
        for seed in [42, 137, 2024]:
            f = os.path.join(base, size, "{}_seed{}".format(lang, seed), "result.json")
            d = json.load(open(f))
            vb = d["val_bpc"]
            tb = d["test_bpc"]["test_bpc"]
            gr = d["grammar"]["grammaticality_rate"] * 100
            bs = d["babi"]["test_seen"]["overall"]["accuracy"] * 100
            bu = d["babi"]["test_unseen"]["overall"]["accuracy"] * 100
            print("{:<6} {:<8} {:<5} {:>7.3f} {:>7.3f} {:>5.1f}% {:>6.1f}% {:>6.1f}%".format(
                size, lang, seed, vb, tb, gr, bs, bu))
    print()
