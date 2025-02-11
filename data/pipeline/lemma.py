import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet


def pos_tagger(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def wordlemma(sent):
    try:
        lemmatizer = WordNetLemmatizer()
        pos_tagged = nltk.pos_tag(nltk.word_tokenize(sent))
        lemmatized_sentence = []
        for word, tag in pos_tagged:
            tag = pos_tagger(tag)
            if tag:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word))
        return lemmatized_sentence[0]
    except:
        return None