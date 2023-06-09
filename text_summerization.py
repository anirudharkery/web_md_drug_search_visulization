import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from heapq import nlargest

nltk.download('punkt')
nltk.download('stopwords')


def summarize_sentences(sentences, num_sentences=3):
    # Tokenize the sentences
    tokenized_sentences = [sentence.lower() for sentence in sentences]
    tokenized_sentences = [sent_tokenize(sentence) for sentence in tokenized_sentences]
    tokenized_sentences = [sentence for sublist in tokenized_sentences for sentence in sublist]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in tokenized_sentences]
    tokenized_sentences = [[word for word in sentence if word.lower() not in stop_words] for sentence in tokenized_sentences]

    # Calculate word frequency
    word_frequencies = FreqDist([word for sentence in tokenized_sentences for word in sentence])

    # Assign scores to sentences based on word frequency
    sentence_scores = {}
    for sentence in tokenized_sentences:
        sentence_str = " ".join(sentence)  # Convert the list of words to a string
        if sentence_str not in sentence_scores.keys():
            sentence_scores[sentence_str] = 0  # Initialize the score
        for word in sentence:
            if word in word_frequencies.keys():
                sentence_scores[sentence_str] += word_frequencies[word]

    # Get the top sentences with the highest scores
    summarized_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summarized_sentences)

    return summary



