# English-Word-2-Vector

We know about mathematical vector.Vector actually stores array of number.Every image have a pixel.We can represent those pixel as a vector.
Same as we find way to represent word into vector.What happens when we represent word to vector. We know in machine learning works with actually numerical value Which is actually vector.If we convert word into vector we can easily train our  million of data(numeriacl data) within few minutes or few hours(it depends on computer capability) 

Word2Vec is a group of models which helps derive relations between a word and its contextual words.

How can we build simple, scalable, fast to train models which can run over billions of words that will produce exceedingly good word representations? You can get this answer here.
https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817



We try to build English word to vector.We follow this pipeline for this
* Collect data
* Sentence tokenization
* Removing punctuation
* Detect common phrases
* Removing Stopword
* Lemmatization
* Training

## Sentence tokenization
In sentence tokenization ,we first separate our data in sentence.Actually our data exist this form.
```
Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.
```
We separate those data in sentence.They have some boundary condition to separate in sentence.Suppose we find a '.'(full stop),'" "'(invaded coma) sign to separate the data to sentence.

Here,we use NLTK(Natural Language Toolkit) python library.This library automatically tokanize the sentence. 

```
import nltk
text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)
```
### After Sentence tokenization
![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/tok2.jpg)

## Removing punctuation
In sentence ,they have some unused punctuation.

Example:Before removing unused punctuation
```
Backgammon is one of the oldest ''known board games. Its history can be', traced back nearly 5,000 years to ;;archeological discoveries"; in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.
```
### After removing punctuation

![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/re.jpg)

We use this code to remove the punctuation

```
from nltk.tokenize import RegexpTokenizer
tt=""

sentences2=[]
for x in sentences:
    tokenizer = RegexpTokenizer(r'\w+')
    text2=tokenizer.tokenize(x)
    cnt=1
    for x2 in text2:
        if cnt==1:
            tt+=x2
            cnt=0
        tt+=" "+x2    
    sentences2.append(tt)
    tt=""
        

```

## Removing Stopword
They have some word which actually do not affect our sentence meaning.

Ex: Some stopwords in English
```
{'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't"}
```
We remove those word from sentence.We use NLTK library which have stop word set.Using those set we can remove stop words.

### Before removing stopword
```
['Backgammon is one of the oldest known board games',
 'Its history can be traced back nearly 5 000 years to archeological discoveries in the Middle East',
 'It is a two player game where each player has fifteen checkers which move between twenty four points according to the roll of two dice']
```

### After removing stopword

![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/stopwords.jpg)

### This is the code
```
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
m=[]
for sentence in sentences2:
    words = nltk.word_tokenize(sentence)
    without_stop_words = [word for word in words if not word in stop_words]
    m.append(without_stop_words)

```


## Lemmatization

In this part ,we convert our word into base word. Suppose,we can write 'do' word in different form like does,done,doing.It is inefficient to write vector for every word.Because we have huge number of word.suppose we have 3 million word,it is difficult for us to map 3 million word as a vector. 
We use WordNetLemmatizer in NLTK library which help you to convert any word in base word.
### Before Lemmatization
```
[['Backgammon', 'one', 'oldest', 'known', 'board', 'games'],
 ['Its',
  'history',
  'traced',
  'back',
  'nearly',
  '5',
  '000',
  'years',
  'archeological',
  'discoveries',
  'Middle',
  'East'],
 ['It',
  'two',
  'player',
  'game',
  'player',
  'fifteen',
  'checkers',
  'move',
  'twenty',
  'four',
  'points',
  'according',
  'roll',
  'two',
  'dice']]
  
```
### After Lemmatization

![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/lem.jpg)

This is the code
```
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
mm=[]
for x in m:
    m2=[]
    for x2 in x:
        lemmatizer = WordNetLemmatizer()
        x3=lemmatizer.lemmatize(x2, wordnet.VERB)
        m2.append(x3)
    mm.append(m2)
```


## Training 
Here,we use genism library to use skim gram,CBOW algoritm.

We take a temporary file "word2vec.model"
```
path = get_tmpfile("word2vec.model")
```
We initialize our model.We use some parameter

1st parameter: refers the text file

2nd parameter: Size of the vector.Ex: They have size 150 which means our vector size is 150.To represent any word we need 150 size array.

3rd parameter: size of the window
They have a picture we see first window take 3 word and 2nd window take 4 word which actually defines window size.

![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/pic.jpg)

4th parameter: min_count means ninimium frequency count of words.

5th parameter: workers means how many threads to use behind the scenes


```
model = Word2Vec(mm,size=10,window=10, min_count=1,workers=13)
```
Then we save our model

```
model.save("word2vec.model")
```
```

from gensim.models import Word2Vec
# train model
model = Word2Vec(mm,size=10,window=10, min_count=1,workers=13)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['know'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)

```
Now we check word 2 vector representation of any word.
```
vector = model.wv['know']
```

Find the similar word of 'know' word.
```
w1=['know']
model.wv.most_similar(positive=w1)
```

![alt text](https://github.com/shahidul034/English-Word-2-Vector/blob/master/Pic/training.jpg)

You can see whole code in this link

### https://www.kaggle.com/shahidul034/english-word2vec
## Reference:
http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XIpNgygzZEZ

https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

https://www.kaggle.com/shahidul034

https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817

https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63?fbclid=IwAR1L20jzZ_cELDxN6zxShg5N_zFReoO5OsFLjQ27qLwjobI7oLRPPiDVdEQ




