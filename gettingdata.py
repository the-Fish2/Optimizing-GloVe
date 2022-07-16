import os

if not {"word2vec.model", "word2vec.model.vectors.npy"}.issubset(set(os.listdir())):
    import gensim.downloader
    wv = gensim.downloader.load("word2vec-google-news-300")
    wv.save("./word2vec.model")

print("File Exists")

from gensim.models import KeyedVectors
wv = KeyedVectors.load("word2vec.model", mmap="r")
print(wv)
