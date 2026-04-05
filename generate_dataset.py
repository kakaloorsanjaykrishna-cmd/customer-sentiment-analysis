import random
import pandas as pd

reviews = [
    ("The product is good but delivery was late", 2),
    ("Amazing quality but packaging was terrible", 2),
    ("Worst experience, but customer support helped", 2),
    ("I love the design but performance is poor", 2),
    ("Battery life is great but camera is bad", 2),

    ("Absolutely fantastic product, loved it!", 1),
    ("Very happy with the purchase", 1),
    ("Works perfectly and fast delivery", 1),

    ("Very disappointed, waste of money", 0),
    ("Poor quality and bad experience", 0),
    ("Stopped working after one day", 0),

    ("It’s okay, nothing special", 2),
    ("Average performance", 2),
    ("Decent product for the price", 2),
]

data = []

for _ in range(5000):
    review, sentiment = random.choice(reviews)

    # add noise
    noise = random.choice([
        "", "!", "!!", "??", " really", " very", " not", " maybe", " slightly"
    ])

    data.append([review + noise, sentiment])

df = pd.DataFrame(data, columns=["review", "sentiment"])
df.to_csv("data/raw/reviews.csv", index=False)

print("🔥 Realistic dataset generated")