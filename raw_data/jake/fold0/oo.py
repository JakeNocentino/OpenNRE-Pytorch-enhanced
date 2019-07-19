import json

with open('data_complete.json', 'r') as infile:
	hi = json.load(infile)

print(hi[1])