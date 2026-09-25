import duckdb
import sys
from sentence_transformers import SentenceTransformer
import json

DB = duckdb.connect("iconclass.duckdb")
DB.execute("INSTALL vss; LOAD vss;")
DB.execute("create table if not exists gemma (notation TEXT, vec FLOAT[512])")

model = SentenceTransformer("google/embeddinggemma-300m", truncate_dim=512)


F = "./data/captions.json"

C = json.load(open(F))

ALL = {}
for o in C:
    caption = o["caption"]
    suggestions = set()
    sentences = [x for x in caption.split(".")]
    sentences.append(caption)
    for sentence in sentences:
        query_vec = model.encode(sentence)
        for notation in DB.execute(
            "SELECT notation FROM gemma ORDER BY vec <=> ? LIMIT 5", [query_vec]
        ).fetchall():
            if notation[0] == "ICONCLASS":
                continue
            suggestions.add(notation[0])
    ALL[o["s"]] = list(suggestions)
    print(o["s"], end="  ")
open("candidates.json", "w").write(json.dumps(ALL))
