import sys

import numpy as np

from sklearn import naive_bayes
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

if len(sys.argv) < 2 or len(sys.argv) > 2:
	print("Como usar:")
	print("$ python %s 'English phrase with a positive or negative sentiment'" % sys.argv[0])
	print("$ python %s 'its a bad film'" % sys.argv[0])
	print("$ python %s 'great!!'" % sys.argv[0])
	print("$ python %s 'negative sentence'" % sys.argv[0])
	sys.exit(1)
else:
	testInput = sys.argv[1]

allReviews = []

# Carregando os reviews de sentimentos negativos
with open("rt-polarity.neg", 'r') as file:
	for line in file:
		allReviews.append([line, 0])

# Carregando os reviews de sentimentos positivos
with open("rt-polarity.pos", 'r') as file:
	for line in file:
		allReviews.append([line, 1])

# randomizar entrada
np.random.shuffle(allReviews) 

trainReviews = allReviews[:int(len(allReviews)*0.9)] # separa 90% para treinamento
testReviews =  allReviews[int(len(allReviews)*0.9):] # separa os 10% restantes para teste

# auxilia na tarefa de transformar entrada crua em bag of words de forma simples, ha varias opcoes para essa tarefa
# a inicializacao esta ignorando palavras que ocorrem apenas uma vez e ignora as chamadas 'stop words', que sao basicamente neutras para uma categorizacao desse tipo
vectorizer = CountVectorizer(stop_words='english', strip_accents='unicode', min_df = 2)

train_features = vectorizer.fit_transform([r[0] for r in trainReviews]) 
test_features = vectorizer.transform([r[0] for r in testReviews])      

# treina o modelo usando a distribuicao Multinomial, dizendo ao modelo os valores conhecidos para as observacoes disponiveis
nb = naive_bayes.MultinomialNB()
nb.fit(train_features, [r[1] for r in trainReviews])

# realiza as predicoes usando o conjunto de teste
predictions = nb.predict(test_features)
testLabels = [r[1] for r in testReviews]

# calcula acuracia
acuracy = metrics.accuracy_score(testLabels, predictions)
print("Acuracia do modelo: %0.5f" % acuracy)	

# predicao da frase enviada pela entrada padrao
inputPrediction = nb.predict( vectorizer.transform( [testInput] ) )[0]
if( inputPrediction == 0 ):
	print('Your input (%s) is a negative sentence' % testInput)
else:
	print('Your input (%s) is a positive sentence' % testInput)


