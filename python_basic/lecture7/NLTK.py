import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# nltk.download()

sentence = "Hi . This is Tom . I have many cars ."
sentence = sentence.lower()
print(sentence)
tokens = nltk.word_tokenize(sentence)
print(tokens)
stop = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop]
print(tokens)

porter_stemmer = PorterStemmer()
tokens = [porter_stemmer.stem(token) for token in tokens]
print(tokens)

print(tokens)
text = nltk.Text(tokens)
print(text)
print(len(text.tokens))
# print(len(set(text.tokens)))
# #
for token in text.vocab():
    print(token, text.vocab()[token])

text.plot(5)

# text.count('.')
# text.count('many')
# text.dispersion_plot(['.', 'many'])





