import nltk
import numpy
import re
import itertools


def cleaning(text):
    import numpy as np
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    lem=WordNetLemmatizer()
    txt = str(text)
    txt = re.sub(r"http\S+", "", txt)       #removing hyperlinks
    if len(txt) == 0:
        return 'no text'
    else:
        txt = txt.split()
        index = 0
        for j in range(len(txt)):
            if txt[j][0] == '@':
                index = j
        txt = np.delete(txt, index)
        if len(txt) == 0:
            return 'no text'
        else:
            words = txt[0]
            for k in range(len(txt)-1):
                words+= " " + txt[k+1]
            txt = words
            txt = re.sub(r'[^\w]', ' ', txt)    #removing special symbols
            if len(txt) == 0:
                return 'no text'
            else:
                txt = ''.join(''.join(s)[:2] for _, s in itertools.groupby(txt))
                txt = txt.replace("'", "")
                txt = nltk.tokenize.word_tokenize(txt)
                for j in range(len(txt)):
                    txt[j] = lem.lemmatize(txt[j], "v")
                if len(txt) == 0:
                    return 'no text'
                else:
                    return txt
