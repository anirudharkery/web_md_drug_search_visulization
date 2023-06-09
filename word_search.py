from gensim.models import Word2Vec



# Load the pre-trained Word2Vec model
model = Word2Vec.load('/Volumes/college/web_md_drug_search_visulization/word2vec_model')



def find_closest_word(word):
    # Get the most similar words based on cosine similarity
    similar_words = model.wv.most_similar(positive=[word])
    
    # Extract the closest word
    closest_word = similar_words[0][0]
    
    return closest_word

def word_search(target_word):
    
        return find_closest_word(target_word)
        

