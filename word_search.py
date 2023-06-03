from gensim.models import Word2Vec



# Load the pre-trained Word2Vec model
model = Word2Vec.load('path_to_word2vec_model')



def find_closest_word(word):
    # Get the most similar words based on cosine similarity
    similar_words = model.most_similar(positive=[word])
    
    # Extract the closest word
    closest_word = similar_words[0][0]
    
    return closest_word

def word_search(words,target_word):
        closest_words = []
        for word in words:
            if word != target_word:
                closest_word = find_closest_word(word)
                closest_words.append(closest_word)
            else:
                closest_words.append(word)

        return closest_words
