f = open("iris.data","r")
lines = f.readlines()
f.close()

s = ''
for line in lines:
	sp = line.split(',')
	label = sp.pop().strip()
	sp.insert(0,label)
	s += ','.join(sp)
	s += '\n'


o = open("iris_modified_for_logistic_regression.data","w")
o.write(s)
o.close()
