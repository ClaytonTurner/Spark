f = open("water-treatment.data",'r')
lines = f.readlines()
f.close()

s = ''
for line in lines:
	sp = line.split(',')
	index = 0
	for item in sp:
		if item == '?':
			sp[index] = '0'
		if item == '?\n':
			sp[index] = '0\n'
		index += 1	
	s += ','.join(sp[1:])

o = open("water-treatment-altered.data","w")
o.write(s)
o.close()
