import pandas as pd
import os
import gc
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

DATA_PATH = "data/malaya_fake_news_preprocessed_dataframe.pkl"
SAVE_PATH = "models/doc2vec_model.d2v"

# Maximum samples to keep in memory
MAX_SAMPLES = 5000

print("Loading dataframe...")
df = pd.read_pickle(DATA_PATH)

if len(df) > MAX_SAMPLES:
    print(f"Sampling {MAX_SAMPLES} rows out of {len(df)} to avoid OOM")
    df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)

# Tag documents
print("Tagging documents...")
documents = df["news"].astype(str).tolist()
tagged = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(documents)]

# Clear dataframe to free memory
del df
gc.collect()

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

model = Doc2Vec(
    vector_size=100,
    window=3,
    min_count=2,
    dm=1,
    negative=5,
    workers=1,
    epochs=5,
    seed=42
)

print("Building vocabulary...")
model.build_vocab(tagged)
gc.collect()

print("Training model...")
model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

model.save(SAVE_PATH)
print(f"✔ Doc2Vec saved to {SAVE_PATH}")
