import fasttext
import fasttext.util
import numpy as np


ft = fasttext.load_model('cc.en.300.bin')

def cos_sim(a, b):
    """Takes 2 vectors a, b and returns the cosine similarity according 
    to the definition of the dot product
    (https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html)
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def compare_word(w, words_vectors):
    """
    Compares new word with those in the words vectors dictionary
    
    """
    vec=ft.get_sentence_vector(w)
    return {w1:cos_sim(vec,vec1) for w1,vec1 in words_vectors.items()}


