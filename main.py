from sklearn import tree, model_selection
from scipy import stats
import graphviz
import random
import numpy as np
import itertools

def render_tree(clf):
	data = tree.export_graphviz(clf, out_file=None) 
	graph = graphviz.Source(data)
	graph.render("test")

def run_gridsearch(training,target,classifier,param_grid, fold_number):
	search = model_selection.RandomizedSearchCV(classifier,param_grid,n_iter = 3,cv = fold_number)
	search.fit(training,target)
	#print(search.best_score_)
	#print(search.cv_results_['params'][search.best_index_])
	return search.best_score_, search.cv_results_['params'][search.best_index_]

def probabilistically_classify(classifier,training_data,target):
	prob_weights = classifier.predict_proba(training_data)
	predictions = []
	for instance in prob_weights:
		rand = random.random()
		#print(rand)
		if rand <= instance[0]:
			predictions.append(0)
		else:
			predictions.append(1)
	correct = 0.0
	for prediction,actual in zip(predictions,target):
		if prediction == int(actual):
			correct+=1
	return correct/len(target)



training_data,target = [],[]
with open('train.txt') as training_file:
	for line in training_file:
		training_data.append([feature for feature in line.strip().split('\t')[:-1]])
		target.append(line.strip().split('\t')[-1])

training_data = np.array(training_data)
target = np.array(target)

clf = tree.DecisionTreeClassifier()

clf.criterion = "gini"
clf.min_samples_split = 32
clf.max_depth = 7
clf.min_samples_leaf = 5
clf.max_features = 67
clf.fit(training_data,target)
predictions = probabilistically_classify(clf,training_data,target)
print(predictions)
print(clf.score(training_data,target))
scores = model_selection.cross_val_score(clf,training_data,target,cv=10)
print(sum(scores)/len(scores))
render_tree(clf)


"""
param_grid = {
	"criterion" : ["gini","entropy"],
	"min_samples_split" : stats.randint(10,108),
	"max_depth" : stats.randint(3,20),
	"min_samples_leaf" : stats.randint(1,10),
	"max_features" : stats.randint(1,176)
}
maxScore = 0
bestParam = None
while (True):
	score,params = run_gridsearch(training_data,target,clf,param_grid,10)
	print("Run complete\n")
	if score > maxScore:
		maxScore = score
		bestParam = params
		print("Accuracy: {}. Params: {}".format(maxScore,bestParam))
		clf.fit(training_data,target)
		render_tree(clf)
#print("The depth and minImpurityDecrease with the best accuracy is {} with {}%.".format(maxDepth,maxScore))
#Accuracy: 0.72725. Paramters: {'criterion': 'gini', 'max_depth': 8, 'max_features': 67, 'min_samples_leaf': 5, 'min_samples_split': 32}
"""