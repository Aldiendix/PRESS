"""Build a pool of recent arXiv titles (2023+) from the public metadata snapshot on Hugging Face."""
import os, subprocess, pandas as pd, pyarrow.parquet as pq
out = []
for i in range(11):
    p = f"data/arxiv/train-{i:05d}.parquet"
    if not os.path.exists(p):
        subprocess.run(["curl", "-s", "-L", "-m", "1200", "-o", p, f"https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot/resolve/main/data/train-{i:05d}-of-00011.parquet"], check=True)
    t = pq.ParquetFile(p).read(columns=["id", "title", "categories"]).to_pandas()
    t = t[t.id.str.match(r"^2[3-6]\d\d\.")]
    t["title"] = t.title.str.replace(r"\s+", " ", regex=True).str.strip()
    out.append(t); print(i, len(t), flush=True); os.remove(p)
d = pd.concat(out).drop_duplicates("title"); d.to_parquet("data/arxiv/titles_2023plus.parquet"); print(len(d), d.id.str[:2].value_counts().sort_index().to_dict())
