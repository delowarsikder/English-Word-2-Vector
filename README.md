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
Here,we use @nltk(Natural Language Toolkit) python library. 
