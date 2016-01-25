import re
import os
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk import PorterStemmer
#from stemming.porter2 import stem
from porter2 import stem
from nltk import SnowballStemmer
import math
import nltk
import glob
from nltk.tokenize import *
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gv 
import sys
import mlpy
import pydot
# Import pygraph
from pygraph.classes.graph import graph
from pygraph.classes.digraph import digraph
from pygraph.algorithms.searching import breadth_first_search
from pygraph.readwrite.dot import write

# for tag cloud
from pytagcloud import create_tag_image, make_tags
from pytagcloud.lang.counter import get_tag_counts
import time

import json
import numpy as np
from bokeh.plotting import *
from bokeh.objects import HoverTool, ColumnDataSource
from collections import OrderedDict


def getTextFeatures(dirName):

	engStopWords = stopwords.words('english')

	# initilizations:
	ignoreCase = True
	iDoc = 0
	tf = []
	idf = []
	tfidf = []
	nWords = []
	words = []
	allFiles = []
	
	for subdir, dirs, files in os.walk(dirName):
	    files.sort() 
	    for file in files:  							# for each file in the given directory:
		file = dirName + file							# update the list of files analyzed:
		allFiles.append(file)

		# print repr(iDoc) + " of " + repr(len(files)) + file
		nWords.append(0) 							# add count of total words in the current document
		for line in open(file): 						# for each file in the current document:
			if ignoreCase: line = line.lower()
			tokenizer = RegexpTokenizer('[\d\.]+|\w+|\$[\d\.]+')		# initialize tokenizer
			tokens = tokenizer.tokenize(line)
			
			for word in tokens: # for each word:
				if len(word) > 2 and word not in engStopWords: 		# if the word is not in the list of stop words and its length is at least 3
					#stemmer = SnowballStemmer("german") 		# TODO: other languages (language detection). Use SnowballStemmer.languages to see list of languages
					#word = stemmer.stem(word)					
					#word = PorterStemmer().stem_word((word))
					word = stem(word)
					# word = WordNetLemmatizer().lemmatize(word, 'v') # stemming

					nWords[iDoc] += 1; 				# update the number of total words in the document
					if word not in words:				# if the current word is not in the GLOBAL list of words (bag of words):
						tf.append([0]*len(files))		# add zeros to the tf matrix
						tfidf.append([0]*len(files))		# add zeros to the tf-idf matrix
						tf[-1][iDoc] += 1			# increase by 1 the tf value of the current element for the current word
						words.append(word)			# add the word to the bag of words
						idf.append(0)				# add a zero to the idf array
					else:
						idxWord = words.index(word)		# find the index of the current word in the bag of words
						tf[idxWord][iDoc] += 1			# update the term frequency matrix
		iDoc = iDoc + 1 							# current number of processed documents:

	numOfDocs = float(iDoc); 							# total number of processed docs

	# post process: compute the final tf array and compute the idf counter:
	for i in range(len(tf)): 							# for each word
		for j in range(len(tf[i])): 						# for each document
			if tf[i][j] > 0:						# if the currect word has appearred (at least once) in the current document:
				idf[i] += 1.0						# update the idf counter
			if (nWords[j] > 0):
				tf[i][j] = tf[i][j] / float(nWords[j])			# normalize the tf value
			else:
				tf[i][j] = 0.0

	T1 = 1.0
	idfTemp = []
	tfTemp = []
	wordsTemp = []
	tfidfTemp = []
	for i in range(len(idf)):
		if idf[i]>T1:
			idfTemp.append(idf[i])
			tfTemp.append(tf[i])
			wordsTemp.append(words[i])
			tfidfTemp.append(tfidf[i])

	idf = list(idfTemp)
	tf = list(tfTemp)
	words = list(wordsTemp)
	tfidf = list(tfidfTemp)

	dFreq = []

	# compute the final tf value
	for i in range(len(idf)):
		dFreq.append(idf[i] / numOfDocs)
		idf[i] = 1.0 + math.log10(numOfDocs/idf[i])
	
	# compute the tf-idf value:
	for i in range(len(tf)): 							# for each word
		for j in range(len(tf[i])): 						# for each document
			tfidf[i][j] = idf[i] * tf[i][j]

	return (allFiles,words, tf, idf, tfidf, dFreq)

def ngrams(word_list, n):

	ngrams = dict()

	#create an n-gram list
	for i in range(len(word_list) - n + 1):
            gram = " ".join([str(num) for num in word_list[i:i+n]])	 
	    if gram in ngrams:
		ngrams[gram] += 1
	    else:
		ngrams[gram] = 1

	#now ngrams contains all the ngrams of the book
	# sorted_ngrams = sorted(ngrams.iteritems(), key = operator.itemgetter(1), reverse = True)

        return ngrams

def getTextFeatures2(dirName):

	engStopWords = stopwords.words('english')
	mystopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
	mergedStopWords = mergedlist = list(set(engStopWords + mystopwords))	
	print len(engStopWords), len(mystopwords), len(mergedStopWords)
	# initilizations:
	ignoreCase = True
	iDoc = 0
	tf = []
	idf = []
	tfidf = []
	nWords = []
	words = []
	allFiles = []

	for subdir, dirs, files in os.walk(dirName):
	    files.sort() 
	    for file in files:  							# for each file in the given directory:
		file = dirName + file							# update the list of files analyzed:
		allFiles.append(file)
		curWords = []
		# print repr(iDoc) + " of " + repr(len(files)) + file
		nWords.append(0) 							# add count of total words in the current document
		for line in open(file): 						# for each file in the current document:
			if ignoreCase: line = line.lower()
			tokenizer = RegexpTokenizer('[\d\.]+|\w+|\$[\d\.]+')		# initialize tokenizer
			tokens = tokenizer.tokenize(line)
			
			for word in tokens: # for each word:
				if len(word) > 2 and not word.isdigit() and word not in mergedStopWords: 	# if the word is not in the list of stop words and its length is at least 3
					# word = WordNetLemmatizer().lemmatize(word, 'v') # stemming
					# word = PorterStemmer().stem_word((word))
					word = stem(word)
					if word.isdigit():
						continue
					nWords[iDoc] += 1; 				# update the number of total words in the document
					curWords.append(word)
					
					if word not in words:				# if the current word is not in the GLOBAL list of words (bag of words):
						tf.append([0]*len(files))		# add zeros to the tf matrix
						tfidf.append([0]*len(files))		# add zeros to the tf-idf matrix
						tf[-1][iDoc] += 1			# increase by 1 the tf value of the current element for the current word
						words.append(word)			# add the word to the bag of words
						#print len(words)
						idf.append(0)				# add a zero to the idf array
					else:
						idxWord = words.index(word)		# find the index of the current word in the bag of words
						tf[idxWord][iDoc] += 1			# update the term frequency matrix
			
		nGrams2 = ngrams(curWords, 2)
		for ngram, count in nGrams2.iteritems():
			if count>1:
				if ngram not in words:				# if the current word is not in the GLOBAL list of words (bag of words):
					tf.append([0]*len(files))		# add zeros to the tf matrix
					tfidf.append([0]*len(files))		# add zeros to the tf-idf matrix
					tf[-1][iDoc] += count			# increase by 1 the tf value of the current element for the current word
					words.append(ngram)			# add the word to the bag of words	
					#print "NGRAM: " + str(len(words))
					idf.append(0)				# add a zero to the idf array
				else:
					idxWord = words.index(ngram)		# find the index of the current word in the bag of words
					tf[idxWord][iDoc] += 1			# update the term frequency matrix

			
		iDoc = iDoc + 1 						# current number of processed documents:
		
		# CLEAR NOT FREQUENT VALUES:
#		if iDoc % 10 == 0:
#			print "DOC " + str(iDoc) + " of " + str(len(files)) + " - - - # WORDS " + str(len(words))

		# BUG!!!!!!!!!!!!!!!!!!!!
		if (iDoc % 500 == 0) | (iDoc == len(files)):
			toRemove = []
			print  "            CLEANING: DOC " + str(iDoc) + " of " + str(len(files)) + " - - - - - Words before cleaning" , len(words), 
			for w in range(len(words)):
				countDocs = sum(x > 0 for x in tf[w])
				if countDocs < 3:
					toRemove.append(w)

			for rindex in sorted(toRemove, reverse=True):
			    	del tf[rindex]
				del tfidf[rindex]
				del idf[rindex]
				del words[rindex]

			print " Words after cleaning" , len(words)
	
	numOfDocs = float(iDoc); 							# total number of processed docs

	# post process: compute the final tf array and compute the idf counter:
	for i in range(len(tf)): 							# for each word
		for j in range(len(tf[i])): 						# for each document
			if tf[i][j] > 0:						# if the currect word has appearred (at least once) in the current document:
				idf[i] += 1.0						# update the idf counter
			if (nWords[j] > 0):
				tf[i][j] = tf[i][j] / float(nWords[j])			# normalize the tf value
			else:
				tf[i][j] = 0.0

	T1 = 1.0

	for i in range(len(idf)):
		if idf[i]<T1:
			idf.pop(i)
			tf.pop(i)
			words.pop(i)
			tfidf.pop(i)

	dFreq = []

	# compute the final tf value
	for i in range(len(idf)):
		dFreq.append(idf[i] / numOfDocs)
		idf[i] = 1.0 + math.log10(numOfDocs/idf[i])
	
	# compute the tf-idf value:
	for i in range(len(tf)): 							# for each word
		for j in range(len(tf[i])): 						# for each document
			tfidf[i][j] = idf[i] * tf[i][j]

	return (allFiles, words, tf, idf, tfidf, dFreq)


def getTextFeatures2_notfidf(dirName):

	engStopWords = stopwords.words('english')
	mystopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
	mergedStopWords = mergedlist = list(set(engStopWords + mystopwords))	

	# initilizations:
	ignoreCase = True
	iDoc = 0
	nWords = []
	words = []
	allFiles = []
	dFreq = []

	for subdir, dirs, files in os.walk(dirName):
	    files.sort() 
	    for file in files:  							# for each file in the given directory:
		#if len(allFiles)>100:
		#	break;
		file = dirName + file							# update the list of files analyzed:
		allFiles.append(file)
		curWords = []
		# print repr(iDoc) + " of " + repr(len(files)) + file
		nWords.append(0) 							# add count of total words in the current document
		statinfo = os.stat(file)
		Size = statinfo.st_size
		if Size<500:
			continue
		if Size>550:
			continue
        
		for line in open(file): 						# for each file in the current document:
			if ignoreCase: line = line.lower()
			if Size<500:
				print Size
			if Size>3000:
				print Size

			tokenizer = RegexpTokenizer('[\d\.]+|\w+|\$[\d\.]+')		# initialize tokenizer
			tokens = tokenizer.tokenize(line)
			
			for word in tokens: # for each word:
				if len(word) > 2 and not word.isdigit() and word not in mergedStopWords: 	# if the word is not in the list of stop words and its length is at least 3
					# word = WordNetLemmatizer().lemmatize(word, 'v') # stemming
					# word = PorterStemmer().stem_word((word))
					word = stem(word)
					if word.isdigit():
						continue
					nWords[iDoc] += 1; 				# update the number of total words in the document
					
					if (word not in words):				# if the current word is not in the GLOBAL list of words (bag of words):
						words.append(word)			# add the word to the bag of words
						dFreq.append(1.0)				# add 1 to the dFreq array
					else:
						if (word not in curWords):			# current word is not in the list of ALREADY READ words of the current doc
							idxWord = words.index(word)		# find the index of the current word in the bag of words
							dFreq[idxWord] += 1.0			# update the dFreq matrix				
					curWords.append(word)

			
		nGrams2 = ngrams(curWords, 2)
		for ngram, count in nGrams2.iteritems():	
			if count>5:
				if ngram not in words:				# if the current word is not in the GLOBAL list of words (bag of words):
					words.append(ngram)			# add the word to the bag of words	
					#print "NGRAM: " + str(len(words))
					dFreq.append(1.0)			# add '1.0' to the dFreq array
				else:
					idxWord = words.index(ngram)		# find the index of the current word in the bag of words
					dFreq[idxWord] += 1.0			# update the freq
			
		iDoc = iDoc + 1 						# current number of processed documents:	
	#	print words, dFreq
	#	raw_input("Press ENTER to exit")
		
	numOfDocs = float(iDoc); 						# total number of processed docs

	dFreq2 = []
	words2 = []

	for i,d in enumerate(dFreq):
		if d>1:
			dFreq2.append(d)
			words2.append(words[i])
	
	dFreq = dFreq2
	words = words2

	# compute the final df value
	for i in range(len(dFreq)):
		dFreq[i] = (dFreq[i] / numOfDocs)

	print len(dFreq)
	return (allFiles, words, dFreq)


def printResults(words, tf, idf,tfidf):
	for x in range(len(words)):
		print '{0:50}'.format(words[x]),
		for y in range(len(tf[x])):
			print '{0:10f}'.format(tfidf[x][y]),
		print

def printMostFrequentWords(words, freq):

	wordsFinal = []
	freqsFinal = []

	for x in range(len(words)):
		if freq[x] > 0.0010:
			wordsFinal.append(words[x])
			freqsFinal.append(freq[x])

	return (wordsFinal, freqsFinal)

def computeSimilarityMatrix(files, tfidf):
	numOfDocs = len(files)
	a = array(tfidf)
	SM = zeros((numOfDocs, numOfDocs))
	for i in range(numOfDocs):
			for j in range(numOfDocs):
				SM[i][j] = float(nltk.cluster.util.cosine_distance(a[:,i],a[:,j]))
	return (SM)

def drawGraphFromSM2(SM, names, outFile, Cut):
	graph = pydot.Dot(graph_type='graph')

	# THRESHOLD SM:
	nonZeroMean = np.mean(SM[SM.nonzero()])
	if Cut:
		T = 5.0 * nonZeroMean
	else:
		T = 0.0;

	for i in range(SM.shape[0]):
		for j in range(SM.shape[0]):
			if SM[i,j] <= T:
				SM[i,j] = 0.0
			else:
				SM[i,j] = 1.0

	numOfConnections = sum(SM, axis = 0)
	#fig = plt.figure(1)
	#plot1 = plt.imshow(SM, origin='upper', cmap=cm.gray, interpolation='nearest')
	#plt.show()

	numOfConnections = 9*numOfConnections / max(numOfConnections)

	for i,f in enumerate(names):	
		if sum(SM[i,:])>0:
			fillColorCurrent = "{0:d}".format(int(ceil(numOfConnections[i])))
			# NOTE: SEE http://www.graphviz.org/doc/info/colors.html for color schemes
			node = pydot.Node(f, style="filled", fontsize="8", shape="egg", fillcolor=fillColorCurrent, colorscheme = "reds9")
			graph.add_node(node)
			
	for i in range(len(names)):
		for j in range(len(names)):
			if i<j:
				if SM[i][j] > 0:
					#gr.add_edge((names[i], names[j]))				
					edge = pydot.Edge(names[i], names[j])	
					graph.add_edge(edge)
	graph.write_png(outFile)

def drawGraphFromSM2_D3js(SM, names, outFile):

	# generate data from SM and names:
	data = {'nodes':[], 'links':[]}
	for i in range(SM.shape[0]):
		#curNode = {'group':groupName[i], 'name':names[i]}
		curNode = {'group':1, 'name':names[i]}
		data['nodes'].append(curNode)

	for i in range(SM.shape[0]):		
		for j in range(SM.shape[1]):
			curLink = {'source': i, u'target': j, u'value': SM[i][j]}
			data['links'].append(curLink)
	nodes = data['nodes']
	names = [node['name'] for node in sorted(data['nodes'], key=lambda x: x['group'])]

	N = len(nodes)
	counts = np.zeros((N, N))
	for link in data['links']:
	    counts[link['source'], link['target']] = link['value']
	    counts[link['target'], link['source']] = link['value']

	colormap = [
	    "#444444", "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
	    "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"
	]

	xname = []
	yname = []
	color = []
	alpha = []
	for i, n1 in enumerate(nodes):
	    for j, n2 in enumerate(nodes):
		xname.append(n1['name'])
		yname.append(n2['name'])

		a = min(counts[i,j]/4.0, 0.9) + 0.1
		alpha.append(a)

		if n1['group'] == n2['group']:
		    color.append(colormap[n1['group']])
		else:
		    color.append('lightgrey')

	output_file(outFile)

	source = ColumnDataSource(
	    data=dict(
		xname=xname,
		yname=yname,
		colors=color,
		alphas=alpha,
		count=counts.flatten(),
	    )
	)

	figure()

	rect('xname', 'yname', 0.9, 0.9, source=source,
	     x_range=list(reversed(names)), y_range=names,
	     color='colors', alpha='alphas', line_color=None,
	     tools="resize,hover,previewsave", title="Similarities",
	     plot_width=800, plot_height=800)

	grid().grid_line_color = None
	axis().axis_line_color = None
	axis().major_tick_line_color = None
	axis().major_label_text_font_size = "5pt"
	axis().major_label_standoff = 0
	xaxis().location = "top"
	xaxis().major_label_orientation = np.pi/3

	hover = [t for t in curplot().tools if isinstance(t, HoverTool)][0]
	hover.tooltips = OrderedDict([
	    ('names', '@yname, @xname'),
	    ('count', '@count'),
	])

	show()      # show the plot



def drawGraphFromSM(SM, names, outFile):
	fig = plt.figure(1)
	plot1 = plt.imshow(SM, origin='upper', cmap=cm.gray, interpolation='nearest')
	plt.show()
		
	gr = graph()

	namesNew = []
	for i,f in enumerate(names):	
		if sum(SM[i,:])>0:
			gr.add_nodes([f])
			namesNew.append(f)
			
	Max = SM.max()
	Mean = mean(SM)

	Threshold = Mean * 1.5
	for i in range(len(names)):
		for j in range(len(names)):
			if i<j:
				if SM[i][j] > 0:
					gr.add_edge((names[i], names[j]))
	# Draw as PNG
	dot = write(gr)
	gvv = gv.readstring(dot)
	gv.layout(gvv,'dot')
	gv.render(gvv,'png', outFile)


def loadDictionaries(dirName):

	dictionaries = []
	dictionariesWeights = []
	dictionariesNames = []

	#for subdir, dirs, files in os.walk(dirName):
	for file in glob.glob(dirName + "/*"):
		dictionariesNames.append(os.path.basename(file))
		dictionaries.append([])
		dictionariesWeights.append([])
		#file = dirName + file			
		for line in open(file): 		
			splits = line.split("\t")
			if (len(splits)==2):
				dictionaries[-1].append(splits[0])
				dictionariesWeights[-1].append(float(splits[1].replace("\n","")))

	return (dictionariesNames, dictionaries, dictionariesWeights)


def generateTagCloudFromWordFreqs(Words, Freqs, outputFile):
	# example: generateTagCloudFromWordFreqs(dictionaries[110][:30], dictionariesWeights[110][:30],'a.png')
	wordFreqs = zip(Words, Freqs)
	tags = make_tags(wordFreqs, maxsize=50)
	create_tag_image(tags, outputFile, size=(600, 500), fontname='Lobster')


def classifyFile(fileName, dictionaries, dictionariesWeight, dictionariesNames, numOfResultsReturned, PLOT_FIGURE):

		
	engStopWords = stopwords.words('english')		
	ignoreCase = True

	nClasses = len(dictionaries)

	Ps  = [0.0] * nClasses
	nWords = 0;
	curWords = []
	curFreqs = []
	totalWords = 0
	tokenizer = RegexpTokenizer('[\d\.]+|\w+|\$[\d\.]+')		# initialize tokenizer
	# STEP A: GENERATE LIST OF WORDS (AFTER STEMMING AND STOPWORD REMOVAL):
	for line in open(fileName): 						# for each file in the current document:
		if ignoreCase: line = line.lower()
		tokens = tokenizer.tokenize(line)	
		
		for word in tokens: # for each word:
			if len(word) > 2 and word not in engStopWords: 		# if the word is not in the list of stop words and its length is at least 3
				#word = WordNetLemmatizer().lemmatize(word, 'v') # stemming
				# word = PorterStemmer().stem_word((word))
				word = stem(word)
				totalWords += 1.0
				if word not in curWords:
					curWords.append(word)
					curFreqs.append(1.0)
				else:
					curFreqs[curWords.index(word)] += 1.0

	normalizeFactor = (totalWords / 15.0)

	# STEP B: PROPABILITY PRODUCT COMPUTATION (BASED ON SINGLE WORDS)
	for iword, word in enumerate(curWords):
		FOUND_word = 0;
		for d in range(len(dictionaries)):
			dic = dictionaries[d]			
			if word in dic:
				idxWord = dic.index(word)
				toMulti = 1.0 + dictionariesWeight[d][idxWord]				
				#Ps[d] *= (toMulti + (1.0 - (1.0/nClasses)))**(curFreqs[iword]/normalizeFactor)	
				#if toMulti>20:
				Ps[d] += math.log(toMulti)								
				FOUND_word = 1;
			else:
				#Ps[d] *= (1.0 - (1.0/nClasses))**(1.0/normalizeFactor)
				Ps[d] += (0.0)
		if (FOUND_word==1):
			#print word
			nWords += 1

	print Ps
	
	# STEP C: PROBABILITY PRODUCT COMPUATION (BASED ON N-GRAMS):
	nGrams2 = ngrams(curWords, 2)
	for ngram, count in nGrams2.iteritems():
		FOUND_word = 0;
		for d in range(len(dictionaries)):
			dic = dictionaries[d]	
			if ngram in dic:
					idxWord = dic.index(ngram)
					toMulti = 1.0 + dictionariesWeight[d][idxWord]
					Ps[d] += math.log(toMulti)
					# Ps[d] *= (toMulti + (1.0 - (1.0/nClasses)))**(1.0/normalizeFactor)
					# print nGram, toMulti
					FOUND_word = 1;
			else:
					#Ps[d] *= (1.0 - (1.0/nClasses))**(1.0/normalizeFactor)
					Ps[d] += 0.0
			if (FOUND_word==1):
				#print word
				nWords += 1
	print Ps
	
	for d in range(nClasses):
		if nWords>0:        
			Ps[d] /= len(curWords)
			# Ps[d] /= (len(dictionariesWeight[d])+0.00000000001)
			# Ps[d] /= nWords
			# Ps[d] /= sum(dictionariesWeight[d])    
		else:
			Ps[d] = 0	
	
	MEANPs = mean(Ps)
	MAX = max(Ps)

	finalLabels = []
	finalLabelsPs = []


	IndecesSorted = [i[0] for i in sorted(enumerate(Ps), key=lambda x:x[1], reverse=True)]
	
	for i in range(numOfResultsReturned):			
		finalLabels.append(dictionariesNames[IndecesSorted[i]])
		#finalLabelsPs.append(Ps[IndecesSorted[i]] / MAX)
		finalLabelsPs.append(Ps[IndecesSorted[i]])

#	for i in range(len(Ps)):
#		if Ps[i] > 2.0 * MEANPs:
#			# print(dictionariesNames[i] + "\t\t\t\t" + str(Ps[i]))
#			finalLabels.append(dictionariesNames[i])
#			finalLabelsPs.append(Ps[i])

	if (PLOT_FIGURE==1):
		fig = plt.figure()
		plt.bar(arange(1,numOfResultsReturned+1)-0.5, array(finalLabelsPs))
		for i in range(numOfResultsReturned):
			plt.text(i+1, 0, finalLabels[i], rotation=90, size=10,horizontalalignment='center',verticalalignment='bottom')
		plt.xticks(range(numOfResultsReturned), [], size='small')
	
		plt.show();

	#print nClasses
	#plt.savefig('new.png', dpi = 500);
	
	return (finalLabels, finalLabelsPs)	


def main(argv):
	if (len(argv)==2):
		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# THIS CODE IS FOR CLUSTERING - FEATURE EXTRACTION:
		# get tf-idf featuers:
		[files,words, tf, idf, tfidf, dFreq]  = getTextFeatures(argv[1])
		# print tf idf values:
		printMostFrequentWords(words, dFreq)
		if len(files)>0:
			SM = computeSimilarityMatrix(files,tfidf)
			nodeNames = []
			for f in files:
				nodeNames.append(os.path.basename(f).replace("http:__arxiv.org_",""))
			drawGraphFromSM(SM, files, 'graph.png')
			cls, means, steps = mlpy.kmeans(SM, k=2, plus=False)
			fig = plt.figure(1)
			plt.plot(cls)
			plt.show()
			


		# # # # # # # # # # # # # # # # # # # # # # # # # # #
		# THIS CODE IS FOR FILE CLASSIFICATION:
	else:
		if (len(argv)==3):			
			[dictionariesNames, dictionaries, dictionariesWeights] = loadDictionaries(argv[1])
			[Labels, LabelsPs] = classifyFile(argv[2], dictionaries, dictionariesWeights, dictionariesNames, 4, 1)
			for i in range(len(Labels)):
				print Labels[i] + "\t\t" + str(LabelsPs[i])
	

if __name__ == '__main__':
	main(sys.argv)
