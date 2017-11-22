from sklearn import tree
import graphviz
import random

training_data,target = [],[]
validation_data,validation_target = [],[]
with open('train.txt') as training_file:
	for line in training_file:
		if random.random() >= .3:
			training_data.append([feature for feature in line.strip().split('\t')[:-1]])
			target.append(line.strip().split('\t')[-1])
		else:
			validation_data.append([feature for feature in line.strip().split('\t')[:-1]])
			validation_target.append(line.strip().split('\t')[-1])

clf = tree.DecisionTreeClassifier()
clf.max_depth = 8
clf.fit(training_data,target)
print(clf.score(validation_data,validation_target))
data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(data)
graph.render("test")