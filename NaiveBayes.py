
import numpy as np 
import pandas as pd
from itertools import combinations
from scipy import  array

def submission(filename,prediction):
	labels = ['id,ACTION']
	for i,p in enumerate(prediction):
		labels.append('%i,%f' %(i+1, p))
	f = open(filename, 'w')
	f.write('\n'.join(labels))
	f.close()
	print 'results saved'


def group_data(data, degree=3):
    """ 
    Groups all columns of data into all combinations of triples
    """
    new_data = []
    m,n = data.shape
    for indicies in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:,indicies]])
    return array(new_data).T


def counts(List):
	"""
	returns dict mapping values to counts 
	"""
	uniques = set(list(List))
	counts = dict((u, np.sum(List==u)) for u in uniques)
	return counts 


class NBClassifier(object):
	def __init__(self, alpha=1.0):
		self.alpha = alpha

	def fit(self,X,y):
		"""
		trains NB classifier 
		"""
		self.pos_counts = [counts(i) for i in X[y==1].T]
		self.neg_counts = [counts(i) for i in X[y==0].T]
		self.total_pos = float(sum(y==1))
		self.total_neg = float(sum(y==0))
		total = self.total_pos + self.total_neg
		self.pos_prior = self.total_pos/total
		self.neg_prior = self.total_neg/total

	def log_pred(self,X):
		"""
		returns posssitive/ negative class ratio probabilities 
		"""
		m,n = X.shape
		alpha = self.alpha
		total_neg = self.total_neg 
		total_pos = self.total_pos
		preds = np.zeros(m)

		for i ,xi in enumerate(X):
			Pxi_neg = np.zeros(n)
			Pxi_pos = np.zeros(n)
			for j,k in enumerate(xi):
				nc = self.neg_counts[j].get(k,0)
				pc = self.pos_counts[j].get(k,0)
				nneg = len(self.neg_counts[j])
				npos = len(self.pos_counts[j])
				### probabilities with laplace smoothing 
				Pxi_neg[j] = (nc + alpha) / (total_neg + alpha*nneg)
				Pxi_pos[j] = (pc + alpha) / (total_pos + alpha*npos)
			### Log of pos/neg class ratios
			preds[i] = np.log(self.pos_prior) + np.sum(np.log(Pxi_pos)) -  \
						np.log(self.neg_prior) - np.sum(np.log(Pxi_neg))
		return preds

	def predict(self, X, cutoff=0):
		preds = self.log_pred(X)
		return (preds >= cutoff).astype(int)


def main(train_file='data/train.csv', test_file='data/test.csv', output_file = 'NB_predict.csv'):
    # Load data
    print 'Loading data'
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    y = np.array(train_data.ACTION)
    X= np.array(train_data.ix[:,1:-1])    
    X_test = np.array(test_data.ix[:,1:-1]) 


    print 'Transforming data'
    X = group_data(X)
    X_test = group_data(X_test)
    model = NBClassifier(alpha = 1e-10)

    print 'Training NB Classifier'
    model.fit(X,y)

    print 'Predicting'
    preds = model.log_pred(X_test)

    submission(output_file, preds)

    return model, X, y, X_test, preds



if __name__=='__main__':
	args = { 'train_file':  'data/train.csv',
	             'test_file':   'data/test.csv',
	             'output_file': 'nb_predict5.csv' }
	model, X, y, X_test, preds = main(**args)









for i ,xi in enumerate(X):
	Pxi_neg = np.zeros(n)
	Pxi_pos = np.zeros(n)
	for j,k in enumerate(xi):
		nc = neg_counts[j].get(k,0)
		pc = pos_counts[j].get(k,0)
		nneg = len(neg_counts[j])
		npos = len(pos_counts[j])
		### probabilities with laplace smoothing 
		Pxi_neg[j] = (nc + alpha) / (total_neg + alpha*nneg)
		Pxi_pos[j] = (pc + alpha) / (total_pos + alpha*npos)

	preds[i] =np.log(pos_prior) + np.sum(np.log(Pxi_pos)) - \
			np.log(neg_prior) - np.sum(np.log(Pxi_neg))
