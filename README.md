# English-Word-2-Vector
We try to build English word to vector.We follow this pipeline for this
1. Collect data
2. Sentence tokenization
3. Removing punctuation
4. Removing Stopword
5. Lemmatization
6. Training

## Sentence tokenization
In sentence tokenization ,we first separate our data in sentence.Actually our data exist this form
```
Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.
```
We separate those data in sentence.They have some boundary to separate in sentence.Suppose we find a '.'(full stop),'" "'(invaded coma) sign to separate the data to sentence 
Here,we use @nltk(Natural Language Toolkit) python library.This library automatically tokanize the sentence. 
```
import nltk
text = "Backgammon is one of the oldest known board games. Its history can be traced back nearly 5,000 years to archeological discoveries in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice."
sentences = nltk.sent_tokenize(text)
for sentence in sentences:
    print(sentence)
```
## Removing punctuation
In sentence ,they have some unused punctuation.
Example:Before removing unused punctuation
```
Backgammon is one of the oldest ''known board games. Its history can be', traced back nearly 5,000 years to ;;archeological discoveries"; in the Middle East. It is a two player game where each player has fifteen checkers which move between twenty-four points according to the roll of two dice.
```
After removing punctuation
```
Backgammon Backgammon is one of the oldest known board games
Its Its history can be traced back nearly 5 000 years to archeological discoveries in the Middle East
It It is a two player game where each player has fifteen checkers which move between twenty four points according to the roll of two dice
```

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

```
We remove those word from sentence.We use NLTK library which have stop word set.Using those set we can remove stop words.
Before removing stopword
```
Backgammon Backgammon is one of the oldest known board games
Its Its history can be traced back nearly 5 000 years to archeological discoveries in the Middle East
It It is a two player game where each player has fifteen checkers which move between twenty four points according to the roll of two dice
```

After removing stopword
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
  'East'],['It',
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
This is the code
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

In this part ,we convert our word into base word. Suppose,we can write 'do' word in different form like does,done,doing.It is unefficient to write vector for every word.Because we have huge number of word. 
We use WordNetLemmatizer in NLTK library Which help you to move in base word.
Before Lemmatization
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
After Lemmatization
```
[['Backgammon', 'one', 'oldest', 'know', 'board', 'game'],
 ['Its',
  'history',
  'trace',
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
  'checker',
  'move',
  'twenty',
  'four',
  'point',
  'accord',
  'roll',
  'two',
  'dice']]
```
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
model = gensim.models.Word2Vec(mm,size=150,window=10,min_count=2,workers=10)
```
Then we save our model

```
model.save("word2vec.model")
```
```

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import gensim

path = get_tmpfile("word2vec.model")

model = gensim.models.Word2Vec(mm,size=150,window=10,min_count=2,workers=10)
model.save("word2vec.model")

```



