"""TODO
UserWarning: Using default key encoder: SHA-1 is *not* collision-resistant. While acceptable for most cache scenarios, a motivated attacker can craft two different payloads that map to the same cache key. If that risk matters in your environment, supply a stronger encoder (e.g. SHA-256 or BLAKE2) via the `key_encoder` argument. If you change the key encoder, consider also creating a new cache, to avoid (the potential for) collisions with existing keys.
  _warn_about_sha1_encoder()
"""
import os
import pandas as pd

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS

#####
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#####

def doc_build(df, column='question'):
    docs = []
    for idx, row in df.iterrows():
        meta = row.to_dict()
        content = meta.pop(column)
        docs.append(
            Document(
                page_content = content,
                metadata = meta
            )
        )
    return docs

def db_builder(docs, db_dir, cache_dir, embed_model_name="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embed_model_name)
    store = LocalFileStore(cache_dir)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )
    db = FAISS.from_documents(docs, cached_embedder)
    db.save_local(db_dir)

if __name__ == '__main__':
    qbank_file_path = os.path.join('data', 'input_source', 'usmle_qbank.csv')
    cache_dir = "data/qbank_embedding/cache/"
    db_dir = "data/qbank_embedding/faiss/"
    df = pd.read_csv(qbank_file_path)

    docs = doc_build(df)
    db_builder(docs, db_dir, cache_dir)

