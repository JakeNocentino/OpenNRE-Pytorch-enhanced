import json

with open('test.json', 'r') as infile:
	train_file = json.load(infile)

with open('train.json', 'r') as infile:
	test_file = json.load(infile)

print(len(train_file))
print(len(test_file))