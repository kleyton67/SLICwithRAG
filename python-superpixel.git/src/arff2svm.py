import sys

def transform(inputfilename, outputfilename):
	fin = open(inputfilename,'r')
	lines = fin.readlines()
	fin.close()
	fout = open(outputfilename,'w')

	beginToRead = False
	for line in lines:
	    if beginToRead == True:
		if len(line) > 5:# not an empty line
		    #read this line
		    dataList = line.split(',')
		    resultLine = ''
		    resultLine += dataList[-1].strip()[7:]
		    resultLine += ' '
		    for i in range(1,len(dataList)-1):
		        resultLine += str(i)
		        resultLine += (":"+dataList[i]+" ")
		    #print(resultLine)
		    fout.write(resultLine+"\n")

	    if line[0:5] == '@data':
		beginToRead = True

	fout.close()
