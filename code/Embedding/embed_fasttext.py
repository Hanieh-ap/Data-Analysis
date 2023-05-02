# from gensim.models import FastText
from multiprocessing import cpu_count
import fasttext


corpus_path = '/home/reza/NLP/gens_fast/enwik9'
save_path = '/home/reza/NLP/gens_fast/'
vecSize = 128
winSize = 5
numWorkers = cpu_count()-10
print(numWorkers)
epochs = 25
minCount = 2
skipGram = True
modelName = f'size{vecSize}_window{winSize}_skipGram{skipGram}_fake.model'

#model = FastText(corpus_file=corpus_path,
#                 vector_size=vecSize,
#                 window=winSize,
#                 min_count=minCount,
#                 workers=numWorkers,
#                 epochs=epochs,
#                 sg=skipGram)
#model.save(save_path + modelName)
#model = fasttext.train_unsupervised(corpus_path, epoch=20 , lr=0.5, thread=numWorkers, minn=2, maxn=5, dim = 100)
model = FastText.load(save_path + modelName)
#model.get_nsimilar_by_word('car', topn=10, restrict_vocab=None)
while True:
	word = input("enter a word")
	if word == "exit":
		break
	print(model.wv.most_similar(word))

