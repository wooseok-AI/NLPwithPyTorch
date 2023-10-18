CHAPTER2_BASIC_NLP_SKILLS
# Brief Look NLP

## Text Tonkenization


```python
import spacy
from nltk import TweetTokenizer

nlp = spacy.load("en_core_web_sm")
text = "Mary, don't splap the green witch"
print([str(token) for token in nlp(text.lower())])

tweet = u"Snow White and the seven Degrees #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower()))
```

    ['mary', ',', 'do', "n't", 'splap', 'the', 'green', 'witch']
    ['snow', 'white', 'and', 'the', 'seven', 'degrees', '#makeamoviecold', '@midnight', ':-)']


## Unigram, Bigram, Trigram, ..., n-gram


```python
def n_grams(text, n):
  '''
  takes token or text, returns a list of n-gram
  '''
  return [text[i:i+n] for i in range(len(text)-n+1)]

cleaned = ["mary", ",", "n't", "slap", "green", "witch", "."]
print(n_grams(cleaned, 3))
```

    [['mary', ',', "n't"], [',', "n't", 'slap'], ["n't", 'slap', 'green'], ['slap', 'green', 'witch'], ['green', 'witch', '.']]


## Lematization


```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(u"he was running late")
for token in doc:
  print("{} -> {}".format(token, token.lemma_))
```

    he -> he
    was -> be
    running -> run
    late -> late


## POS Tagging (Part of Speach)


```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(u"Mary slapped the green witch")
for token in doc:
  print("{} - {}".format(token, token.pos_))
```

    Mary - PROPN
    slapped - VERB
    the - DET
    green - ADJ
    witch - NOUN


## Chunking and Shallow Parsing


```python
nlp = spacy.load("en_core_web_sm")
doc = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
  print("{} - {}".format(chunk, chunk.label_))
```

    Mary - NP
    the green witch - NP


## Other Methods

Dependency Parsing, Constituent Parsing, WordNet...
