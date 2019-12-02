filename = "./data/classes/coco.names"
i = 1
with open(filename,'r') as f:
	with open("./coco.pbtxt",'w') as of:
		for line in f:
			of.write('item {\n  id:')
			of.write(str(i))
			of.write('\n  name: \'')
			of.write(line.rstrip('\n'))
			of.write('\'\n}\n\n')
			i = i + 1