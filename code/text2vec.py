import numpy as np
#from gensim.models import  FastText
import fasttext

class ftxtGenerator:

    def __init__(self, modelPath, vector_size):
        #self.model = FastText.load(modelPath)
        self.model = fasttext.load_model(modelPath)
        self.vector_size = vector_size

    def text2batch(self, text, winSize, return_tokens=False):
        tokens = text.split()
        if len(tokens)==0:
            print(text) 
            return []
        max_l = len(tokens) + winSize-(len(tokens)%winSize)
        # embeddings = np.zeros([max_l,self.vector_size])
        embeddings = np.zeros([winSize,self.vector_size])
        for i in range(min(winSize, len(tokens))):
            embeddings[i,:] = self.model.get_word_vector(tokens[i])
        return embeddings
        #embeddings[:len(tokens),:] = self.model[tokens]
        # batch_size = int(np.ceil(len(tokens)/winSize))
        # batch = np.zeros([batch_size, winSize, self.vector_size])
        # for i in range(batch_size):
        #     batch[i,:,:] = embeddings[i*winSize:(i+1)*winSize]
        # if return_tokens:
        #     return batch, tokens
        # else:
        #     return batch
        

    def tokens2batch(self, tokens, winSize):
        if len(tokens)==0: return None
        max_l = len(tokens) + winSize-(len(tokens)%winSize)
        embeddings = np.zeros([max_l,self.vector_size])
        for i in range(len(tokens)):
        	embeddings[i,:] = self.model.get_word_vector(tokens[i])
        #embeddings[:len(tokens),:] = self.model[tokens]
        batch_size = int(np.ceil(len(tokens)/winSize))
        batch = np.zeros([batch_size, winSize, self.vector_size])
        for i in range(batch_size):
            batch[i,:,:] = embeddings[i*winSize:(i+1)*winSize]
        return batch

