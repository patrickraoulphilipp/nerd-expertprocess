from __future__ import print_function
from rdflib.graph import Graph
from nltk.util import ngrams
from nltk.corpus import stopwords
from SPARQLWrapper import SPARQLWrapper, JSON
from difflib import SequenceMatcher
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import linear_model
from scipy.special import binom
from numpy import linalg as LA
from common import process_args, query_loop, output_nearest
from collections import Counter
from random import random as rndm
from bisect import bisect
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from xml.dom import minidom
from datasketch import MinHash, MinHashLSH
from datetime import date
from time import gmtime, strftime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import coverage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import zero_one_loss
from sklearn.cluster import DBSCAN
from sklearn import metrics
from textblob import TextBlob
import os, shutil
import ast
import subprocess
import commands
import operator
import difflib
import random
import unicodedata
import time
import cPickle
import datetime
import math
import nltk
import twitter
import requests
import numpy as np
import scipy
import itertools
import json
import string
import re
import hashlib
import urllib
import httplib2
import io
import collections
import threading
import time
import unicodedata as ud
import sys
import wvlib
import logging

budget = dict()
bounds = dict()
getRewards = dict()
getRewardsBy = dict()
IndependenceVectors = []
RelatednessVectors = []
RobustnessVectors = []
candidateVectors = dict()
stateVectors = dict()

fittedParameterizedModel = dict()
candidates = dict()
candidateBeliefs = dict()
textid = dict() 
stateText = dict()
stateType = dict()
candidateBeliefsArr = []
beginCandidates = dict()
endCandidates = dict()
text = dict()
texttypes = dict()
annotatedStateText = dict()
hasReward = dict()
tweetid = dict()
textWithEntites = dict()
hasState = dict()
hypos = []
rewards = []
executed = dict()
datapoints = dict()
candidateHorizon = dict()
horizon = dict()
candidatesBy = dict()
allrewards = dict()
stateCandis = dict()

state_dic = dict()
candidateRewards = dict()
stateVectors = dict()
experts_dic = dict()
dependencies = []
globalNeighbors = dict()
minhashsim = dict()
minhashsim1 = dict()
stateEmbSim = dict()
embeddedsim = dict()
wordindex = dict()
neighbors = dict()
vector = dict()
minhashsim = dict()
minhashsim1 = dict()
stateEmbSim = dict()
embeddedsim = dict()
wordindex = dict()
neighbors = dict()
vector = dict()
X_length = []
Y_length = []
X_extra = []
Y_extra = []
X_ling = []
Y_ling = []
Mrev = dict()
M = dict()
Mrev1 = dict()
M1 = dict()
nbrs_length = NearestNeighbors(radius=30)
nbrs_extra = NearestNeighbors(radius=2)
nbrs_ling = NearestNeighbors(radius=0)
mc = 0
mc1 = 1
lsh0 = MinHashLSH(threshold=0.2, num_perm=128)
lsh1 = MinHashLSH(threshold=0.2, num_perm=128)
expert_weight_dic = dict()
p_min = 0.05

Y_theo = dict()
Y_theo["1"] = set()
Y_theo["0.1"] = set()
Y_theo["0"] = set()

globalT = None
globalmode = None
lower_bound = None
dataset_mode = None

experts_dic = dict()
experts_dic["0"] = ["01", "02", "03"]
experts_dic["0.1"] = ["01", "02", "03"]
experts_dic["1"] = ["11", "12", "13"]
metasims_dic = dict()
metasims_dic["0"] = ['StateTextLengthMetaSimilarity', 'StateCandidateWordEmbeddings', 'StateSameLingualType', 'StateTextExtraChars']
metasims_dic["0.1"] =  ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']
metasims_dic["1"] = ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']
bounds = dict()
bounds[('SameLingualType', "0")] = 20
bounds[('CandidateWordEmbeddings', "0")] = 20
bounds[('CandidateTextMetaSimilarity', "0")] = 20
bounds[('StateTextLengthMetaSimilarity', "0")] = 20
bounds[('StateCandidateWordEmbeddings', "0")] = 20
bounds[('StateSameLingualType', "0")] = 20
bounds[('StateTextExtraChars', "0")] = 20
bounds[('SameLingualType', "1")] = 20
bounds[('SameLingualType', "0.1")] = 20
bounds[('CandidateWordEmbeddings', "0.1")] = 20
bounds[('CandidateTextMetaSimilarity', "0.1")] = 20
bounds[('StateTextLengthMetaSimilarity', "0.1")] = 20
bounds[('StateCandidateWordEmbeddings', "0.1")] = 20
bounds[('StateSameLingualType', "0.1")] = 20
bounds[('StateTextExtraChars', "0.1")] = 20
bounds[('SameLingualType', "1")] = 20
bounds[('CandidateWordEmbeddings', "1")] = 20
bounds[('CandidateTextMetaSimilarity', "1")] = 20
bounds[('StateTextLengthMetaSimilarity', "1")] = 20
bounds[('StateCandidateWordEmbeddings', "1")] = 20
bounds[('StateSameLingualType', "1")] = 20
bounds[('StateTextExtraChars', "1")] = 20
behav = dict()
behav["DOI"] = ["Agree", "Perf", "ConfPerf", "IndError"]
behav["DOR"] = ["Perf", "ConfPerf", "IndError"]
behav["DOI"] = ["Agree", "Perf", "IndError"]
behav["DOR"] = ["Perf", "IndError"]

def getAllVecs(candi, typei):
	global candidateVectors
	global stateVectors
	if typei == "candi":
		query = [[candidates[candi]]]
		words = [w for q in query for w in q]
		veci1 = None
		try: 
			veci1 = wv.words_to_vector(words)
		except:
			veci1 = getTweetVector(candidates[candi],1)
		if veci1 != None:
			candidateVectors[candi] = veci1
		return veci1
	if typei == "state":
		veci2 = None
		veci2 = getTweetVector(candidates[candi],1)
		if veci2 != None:
			stateVectors[candi] = veci2
		return veci2	

def getPure(entity):
	if "YAGO" in entity:
		newString = entity.replace("YAGO:", "")
		return newString
	else:
		if "dbpedia" in entity:
			newString = entity.replace("http://dbpedia.org/resource/", "")
			return newString
		else:
			if "wikipedia" in entity:
				newString = entity.replace("https://en.wikipedia.org/wiki/", "")
				return newString

def getDBpediaURI(entity):
	if "dbpedia" in entity:
		return entity
	else:
		if "YAGO" in entity:
			newstring = entity.replace("YAGO:", "http://dbpedia.org/resource/")
			return newstring
		else:
			if "wikipedia" in entity:
				finalresult = entity.replace('https://en.wikipedia.org/wiki/', 'http://dbpedia.org/resource/')
				return finalresult

def doMS(point, h, metasims):
	if 'SameLingualType' in metasims:
		onlineMetasimcomp(point, 'SameLingualType', h, 0.5)
	if 'CandidateTextMetaSimilarity' in metasims:
		onlineMetasimcomp(point, 'CandidateTextMetaSimilarity', h, None)
	if 'CandidateWordEmbeddings' in metasims:
		onlineMetasimcomp(point, 'CandidateWordEmbeddings', h, None)
	if 'StateTextLengthMetaSimilarity' in metasims:
		onlineMetasimcomp(point, 'StateTextLengthMetaSimilarity', h, 40)
	if 'StateCandidateWordEmbeddings' in metasims:
		onlineMetasimcomp(point, 'StateCandidateWordEmbeddings', h, None)
	if 'StateSameLingualType' in metasims:
		onlineMetasimcomp(point, 'StateSameLingualType', h, 4)
	if 'StateTextExtraChars' in metasims:
		onlineMetasimcomp(point, 'StateTextExtraChars', h, 2) 

def onlineMetasimcomp(c1, typei, h, simi):
	global globalNeighbors
	global minhashsim
	global minhashsim1
	global stateEmbSim
	global embeddedsim
	global wordindex
	global neighbors
	global vector
	global X_length
	global X_extra	
	global X_ling
	global Y_length
	global Y_extra
	global Y_ling
	global Mrev
	global M
	global Mrev1
	global M1
	global nbrs_length
	global nbrs_extra
	global nbrs_ling
	global mc
	global mc1
	global lsh0
	found = None
	try:
		found = globalNeighbors[(c1,typei,str(h))]
	except:
		pass
	if found == None:
		h2 = -1
		try:
			h2 = candidateHorizon[c1]
		except KeyError:
			pass
		if str(h2) == str(h):
			mc += 1
			mc1 += 1
			if typei == 'SameLingualType':
				lt = -1
				try:
					lt = getLingualType(stateText[c1], candidates[c1])
				except:
					pass
				vector[c1] = [float(lt)]
				X_ling.append([float(lt)])
				Y_ling.append(c1)
			if typei == 'CandidateTextMetaSimilarity':
				d = []
				for i in range(len(candidates[c1])):
					d.append(candidates[c1][i])
				m = MinHash(num_perm=128)
				for ch in d:
					m.update(ch.encode('utf8'))
				lsh1.insert('m' + str(mc1), m)
				M1[c1] = m	
				indi = 'm' + str(mc1)
				Mrev1[indi] = c1
			if typei == 'StateTextExtraChars':
				vector[c1] = [float(countExtras(stateText[c1]))]
				X_extra.append([float(countExtras(stateText[c1]))])
				Y_extra.append(c1)
			if typei == 'StateSameLingualType':
				lt = []
				lt = getStateLingualType(stateText[c1])
				d = []
				for p in lt:
					d.append(p)
				m = MinHash(num_perm=128)
				for ch in d:
					m.update(ch.encode('utf8'))
				lsh0.insert('m' + str(mc), m)
				M[c1] = m	
				indi = 'm' + str(mc)
				Mrev[indi] = c1
			if typei == 'StateTextLengthMetaSimilarity':
				vector[c1] = [len(stateText[c1])]
				X_length.append([len(stateText[c1])])
				Y_length.append(c1)
		nbrs = None
		X = None
		Y = None
		if typei == "StateTextExtraChars":
			nbrs = nbrs_extra
			X = X_extra
			Y = Y_extra
		if typei == "StateTextLengthMetaSimilarity":
			nbrs = nbrs_length
			X = X_length
			Y = Y_length
		if typei == "SameLingualType":
			nbrs = nbrs_ling
			X = X_ling
			Y = Y_ling
		if (typei == 'SameLingualType' or typei == 'StateTextLengthMetaSimilarity' 
		or typei == 'StateTextExtraChars'):
			nbrs.fit(X)
			h2 = -1
			try:
				h2 = candidateHorizon[c1]
			except KeyError:
				pass
			if str(h2) == str(h):
				ind = nbrs.radius_neighbors([vector[c1]], simi)
				neighbors[c1] = ind
			candiNeighbrs = []
			if not len(neighbors[c1][1]) == 0:
				for it in range(len(neighbors[c1][1][0])):
					no = neighbors[c1][1][0][it]
					candiNeighbrs.append(Y[no])
			globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
		else:
			if typei == "StateSameLingualType":
				h2 = -1
				try:
					h2 = candidateHorizon[c1]
				except KeyError:
					pass
				if str(h2) == str(h):
					fin = []
					nearest = lsh0.query(M[c1])
					for n in nearest:
						fin.append(Mrev[n])
						sim = M[c1].jaccard(M[Mrev[n]])
						minhashsim[(c1,Mrev[n])] = sim
						minhashsim[(Mrev[n],c1)] = sim
					globalNeighbors[(c1,typei,str(h))] = fin
			else:
				if typei == "StateCandidateWordEmbeddings":
					it = 0
					candiNeighbrs = []
					it +=1
					h2 = -1
					try:
						h2 = candidateHorizon[c1]
					except KeyError:
						pass
					if str(h2) == str(h):
						veci1 = None
						try:
							veci1 = stateVectors[c1]
						except:
							veci1 = getAllVecs(c1, "state")
						for c2 in candidates:
							if (textid[c1] != textid[c2] and str(candidateHorizon[c2]) == str(h)
							 and not("_NIL" in candidates[c2])):
								veci2 = None
								try:
									veci2 = stateVectors[c2]
								except:
									veco2 = getAllVecs(c2, "state")
								try:
									v1_norm = veci1/np.linalg.norm(veci1)
									v2_norm = veci2/np.linalg.norm(veci2)	
									numba = np.dot(v1_norm, v2_norm)
									candiNeighbrs.append(c2)
									stateEmbSim[(c1,c2)] = numba
									stateEmbSim[(c2,c1)] = numba
								except:
									pass
						globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
				else:
					if typei == "CandidateWordEmbeddings":
						it = 0
						candiNeighbrs = []
						it +=1
						h2 = -1
						try:
							h2 = candidateHorizon[c1]
						except KeyError:
							pass
						if str(h2) == str(h):
							veci1 = None
							try:
								veci1 = candidateVectors[c1]
							except:
								veci1 = getAllVecs(c1, "candi")
							for c2 in candidates:
								if (textid[c1] != textid[c2] and str(candidateHorizon[c2]) == str(h)
								 and not("_NIL" in candidates[c2])):
									veci2 = None
									try:			
										veci2 = candidateVectors[c2]
									except:
										veci2 = getAllVecs(c2, "candi")
									try:
										v1_norm = veci1/np.linalg.norm(veci1)
										v2_norm = veci2/np.linalg.norm(veci2)	
										res = np.dot(v1_norm, v2_norm)
										candiNeighbrs.append(c2)
										embeddedsim[(c1,c2)] = res
										embeddedsim[(c2,c1)] = res
									except:
										pass
							globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
					else:
						if typei == "CandidateTextMetaSimilarity":
							h2 = -1
							try:
								h2 = candidateHorizon[c1]
							except KeyError:
								pass
							if str(h2) == str(h):
								fin = []
								nearest = lsh1.query(M1[c1])
								for n in nearest:
									fin.append(Mrev1[n])
									sim = M1[c1].jaccard(M1[Mrev1[n]])
									minhashsim1[(c1,Mrev1[n])] = sim
									minhashsim1[(Mrev1[n],c1)] = sim
								globalNeighbors[(c1,typei,str(h))] = fin
		hor = -1
		try:
			hor = candidateHorizon[c1]
		except KeyError:
			hor = 5
		if str(hor) == str(h):
			kkk = []
			avg = 0.0
			newNeighbs = []
			for c2 in globalNeighbors[(c1,typei,str(h))]:
				kkk.append(c2)
				if typei == 'SameLingualType':
					newNeighbs.append([c2, str(1.0)])
				if typei == 'CandidateTextMetaSimilarity':
					simi = minhashsim1[(c1,c2)]
					newNeighbs.append([c2, simi])
				if typei == 'CandidateWordEmbeddings':
					simi = embeddedsim[(c1,c2)]
					newNeighbs.append([c2, simi])
				if typei == 'StateTextExtraChars':
					numba = 0
					numba = float(countExtras(stateText[c1])) / float(countExtras(stateText[c2]))
					if numba > 1:
						numba = 1 / float(numba)
					newNeighbs.append([c2, str(numba)])
				if typei == 'StateSameLingualType':
					simi = minhashsim[(c1,c2)]
					newNeighbs.append([c2, simi])
				if typei == 'StateCandidateWordEmbeddings':
					simi = stateEmbSim[(c1,c2)]
					newNeighbs.append([c2, simi])
				if typei == 'StateTextLengthMetaSimilarity':
					numba = 0
					numba = float(len(stateText[c1])) / float(len(stateText[c2]))
					if numba > 1:
						numba = 1 / float(numba)
					newNeighbs.append([c2,str(numba)]) 
			globalNeighbors[(c1,typei,str(h))] = newNeighbs

def getNeigbs():
	global getRewards
	global getRewardsBy
	for rew in rewards:
		temp = []
		try:					
			temp = getRewards[(rew[2],rew[1])]
			if not(rew in temp):
				temp.append(rew)
		except KeyError:
			temp.append(rew)
		getRewards[(rew[2],rew[1])] = temp
	for rew in rewards:
		temp = []
		try:					
			temp = getRewardsBy[(rew[2],rew[4])]
			if not(rew in temp):
				temp.append(rew)
		except KeyError:
			temp.append(rew)
		getRewardsBy[(rew[2],rew[4])] = temp

def countExtras(tweet):
	counthashtext = tweet.count('#')
	countattext = tweet.count("@")	
	return counthashtext + countattext + 1	

posMapper = dict()

def getWordVector(query, ngram):
	vector = None
	words = [query]
	try:
		vector = wv.words_to_vector(words)
	except:
		if ngram > 1:
			singles = candidateWord.split()
			rx = re.compile('\W+')
			construed = ""
			for s in singles:
				res = rx.sub(' ', s).strip()
				if construed == "":
					construed += s
				else:
					construed = construed + "_" + s
			query = [[construed]]
			words = [w for q in query for w in q]
			vector = wv.words_to_vector(words)
	return vector

def getTweetVector(tweet, ngram):
	count = 0
	currentVector = None
	tokenized = nltk.word_tokenize(tweet)
	filtered_words = [word for word in tokenized if word not in stopwords.words('english')]
	usies = []
	for t in filtered_words:
		if len(t) > 1:
			usies.append(t)
	if ngram == 1:
		for t in filtered_words:
			vector = getWordVector(t, ngram)
			if not(currentVector == None) and not(vector == None):
				currentVector = np.add(currentVector, vector)
				count += 1
			else:
				if currentVector == None and not(vector == None): 
					currentVector = vector
					count += 1
	if not(currentVector == None):
		currentVector = np.divide(currentVector, count)
	return currentVector

def getStateLingualType(tweet):
	res = []
	tokenized = nltk.word_tokenize(tweet)
	pos = nltk.pos_tag(tokenized)	
	blob = TextBlob(tweet)
	for p in pos:
		res.append(p[1])
	return res

def formatDataset(data):
	datapoints = dict()
	if data == 'Microposts':
		with io.open('./data/micro_14.txt', 'r', encoding='utf-8') as f:
			total_ents = 0
			for l in f:
				l = l.encode("ascii", "ignore")
				temp = l.strip().split("\t")
				datapoints[str(temp[0])] = "test"
				datapoints[str(temp[0])] = [temp[1], 'tweet', 'Microposts_2014',[]]
				a = np.array(temp)
				a = np.delete(a, [0,1])
				helper = []
				for item in a:
					helper.append(item)
					if len(helper) == 2:
						pred0 = helper[0]
						pred0 = pred0.decode('unicode_escape')
						pred0 = unicodedata.normalize('NFKD', pred0).encode('ascii','ignore')
						pred1 = helper[1]
						pred1 = pred1.decode('unicode_escape')
						pred1 = unicodedata.normalize('NFKD', pred1).encode('ascii','ignore')
						aggr = [pred0, -1, -1, pred1]
						total_ents += 1		
						datapoints[str(temp[0])][3].append(aggr)
						helper = []
			print("Micro has #tweets=" +str(len(datapoints)) + ", and #ents="+str(total_ents))
	if data == 'Spotlight':
		g = Graph()
		g.parse('./data/spotlight.n3', format='n3')
		qres = g.query(
    		"""Prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
		Prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#>        
		SELECT ?x ?statetext  ?dne ?candidate ?start ?end ?candidatetext
		WHERE {
        	?x ?y nif:Sentence;
		    nif:anchorOf ?statetext.
                ?candidate  nif:sentence       ?x;
			    itsrdf:taIdentRef  ?dne;
			    nif:beginIndex     ?start;
			    nif:endIndex       ?end;
			    nif:anchorOf       ?candidatetext.
       		}""")
		for row in qres:			
			uri = row["x"].encode("ascii", "ignore")
			statetext = row["statetext"].encode("ascii", "ignore")
			candidate = row["candidate"].encode("ascii", "ignore")
			dne = row["dne"].encode("ascii", "ignore")
			start = row["start"].encode("ascii", "ignore")
			end = row["end"].encode("ascii", "ignore")
			candidatetext = row["candidatetext"].encode("ascii", "ignore")
			aggr = [candidatetext, start, end, dne]
			stuff = []
			try:
				stuff = datapoints[uri]
				current = stuff[3]
				current.append(aggr)
				stuff[3] = current
			except KeyError:
				pass
			if len(stuff) == 0:
				datapoints[uri] = [statetext, 'NY Times', 'Spotlight', [aggr]]
	return datapoints

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def newLivedoM(protagonist, sid, cid, metasims, h, threshold, bounds):
	global globalmode
	global globalT
	tweedid = textid[cid]
	is_nil = False
	is_av = False
	metasimarray = []
	flag = 0
	RelatednessVector = None
	gotcha = None
	RelatednessVector = [protagonist,sid,cid,[],globalmode,globalT]
	for metasim in metasims:
		result = []
		computeM = 0
		counter = 0
		c2 = 0
		total = 0
		total2 = 0.0
		ykeeper = 0
		fullcounter = 0
		tobreak = 0
		neighs = []
		try:
			neighs = globalNeighbors[(cid,metasim,str(h))]
		except:
			pass
		for neigh in neighs:
			rews2 = []
			try:
				rews2 = getRewards[(protagonist,neigh[0])]
			except KeyError:
				continue
			for rew2 in rews2:
				is_nil_sim = False
				try:
					if "_NIL" in candidates[rew2[4]]:
				 		is_nil_sim = True
				except:
					is_av = False
				if (rew2[2] == protagonist and sid != rew2[0] and cid != rew2[1] 
				and tweetid != rew2[7]):
					similar = 0
					similar = neigh[1]
					if float(similar) >= float(threshold):
						computeM += float(similar)
						weight = float(similar)
						c2 += 1
						total2 += (float(rew2[6]) * float(weight))
						counter += weight
						fullcounter += 1
						flag = 1
		if not(counter == 0):
			computeM = float(computeM) / float(bounds[metasim, h])
			if computeM > 1.0:
				computeM = 1.0
			res2 = ( float(total2)/float(counter) )
			metasimarray.append([res2, computeM, metasim, protagonist])
			prior = RelatednessVector[3]
			prior.append([metasim,res2, computeM])
			RelatednessVector[3] = prior
		else:
			metasimarray.append([0, 0, metasim, protagonist])
			prior = RelatednessVector[3]
			prior.append([metasim,0, 0])
			RelatednessVector[3] = prior
	RelatednessVectors.append(RelatednessVector)
	return metasimarray

def newLivedoIPairwise(protagonist, competitor, sid, cid, metasims, h, threshold, bounds):	
	global globalmode
	global globalT
	predictors = []
	global IndependenceVectors
	for metasim in metasims:
		agreementRate = []
		carr = []
		performanceRate = []
		confidenceAdjustedPerformanceRate = []
		independentErrorRate = []
		computeM = 0
		ccounter = 0
		try:
			bla =  globalNeighbors[(cid,metasim,str(h))]
		except:
			globalNeighbors[(cid,metasim,str(h))] = []
		for neigh in globalNeighbors[(cid,metasim,str(h))]:	
			rewhelpers = []
			try:
				rewhelpers = getRewards[(protagonist,neigh[0])]
			except KeyError:
				continue
			for rewhelper in rewhelpers: 
				if rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1
				] and tweetid != rewhelper[7]:
					similar = neigh[1]	
					if similar >= threshold:
						rews2 = []
						try:
							rews2 = getRewards[(competitor,neigh[0])]
						except KeyError:
							continue
						for rew2 in rews2:
							if (rew2[0] != sid and rew2[1] == rewhelper[1] and rew2[0] == rewhelper[0]
							 and rew2[2] == competitor):
								computeM += float(similar)
								ccounter += 1
								inderror = abs(float(rewhelper[6]) - float(rew2[6]))
								performanceConf = ( float(rewhelper[6]) + float(rew2[6]) ) / 2
								if rewhelper[4] != rew2[4]:
									agree = 0
									agreeConf = 0
								else:
									agree = 1
									agreeConf = 1.0 - abs(float(rewhelper[5]) - float(rew2[5]))
								carr.append([float(similar) * agree, similar])
								if rewhelper[4] == rew2[4]:
									confidenceAdjustedPerformanceRate.append([float(similar) 
										* float(performanceConf), similar])
								else:
									performanceRate.append([float(similar) * float(rewhelper[6]), similar])
									agreementRate.append([float(similar) * float(rew2[6]), similar])
								if not(rew2[6] > 0.6 and rewhelper[4] > 0.6):
									independentErrorRate.append([float(similar) * float(inderror), similar])
								else:
									independentErrorRate.append([float(similar) * 1.0, similar])									
								break			
		totalagree = 0.0						
		agreecounter = 0.0
		for agree in agreementRate:
			totalagree += float(agree[0])
			agreecounter += float(agree[1])
		agreeresult = 0
		if not(float(agreecounter) == 0.0):
			agreeresult = float(totalagree)/float(agreecounter)
		totalinderror = 0.0
		inderrorcounter = 0.0
		for ind in independentErrorRate:
			totalinderror += float(ind[0])
			inderrorcounter += float(ind[1])
		indresult = 0
		if not(float(inderrorcounter) == 0.0):
			indresult = float(totalinderror)/float(inderrorcounter)
		totalperformance = 0.0
		performancecounter = 0.0
		for perf in performanceRate:
			totalperformance += float(perf[0])
			performancecounter += float(perf[1])
		perfresult = 0
		if not(float(performancecounter) == 0.0):
			perfresult = float(totalperformance)/float(performancecounter)
		totalconfagree = 0.0
		confagreecounter = 0.0
		for confagree in carr:
			totalconfagree += float(confagree[0])
			confagreecounter += float(confagree[1])
		confagreeresult = 0
		if not(float(confagreecounter) == 0.0):
			confagreeresult = float(totalconfagree)/float(confagreecounter)
		confperformance = 0.0
		confperfcounter = 0.0
		for confperf in confidenceAdjustedPerformanceRate:
			confperformance += float(confperf[0])
			confperfcounter += float(confperf[1])
		confperfresult = 0
		if not(float(confperfcounter) == 0.0):
			confperfresult = float(confperformance)/float(confperfcounter)
		norm_m = 0.0
		try:
			norm_m = float(computeM) / float(ccounter)
		except:
			pass
		visited = float(ccounter) / float(bounds[(metasim, h)])
		density = None
		if float(visited) < 1.0:
			density = norm_m * visited
		else:
			density = norm_m
		if float(density) > 1.0:
			density = 1.0	
		IndependenceVectors.append([[protagonist, competitor], sid, cid, metasim, [[agreeresult, "Agree", density]
			, [(1.0 - float(agreeresult)) , "NegConfAgree", density], [confagreeresult, "ConfAgree", density], 
			[confperfresult, "ConfPerf", density], [perfresult, "Perf", density], [indresult, "Error", density]],
			 globalmode, globalT])
		predictors.extend([[agreeresult, density, metasim, protagonist, "Agree", competitor], [confagreeresult,
		density, metasim, protagonist, "ConfAgree", competitor], [confperfresult, density, metasim, protagonist,
		"ConfPerf", competitor], [perfresult, density, metasim, protagonist, "Perf", competitor], [indresult,
		density, metasim, protagonist, "IndError", competitor]])
	return predictors

def newLivedoR(protagonist, competitor, sid, cid, metasims, h, threshold, bounds):
	threshold = 0.5
	predictors = []
	global globalT
	for metasim in metasims:
		performanceRate = []
		confidenceAdjustedPerformanceRate = []
		independentErrorRate = []
		computeM = 0
		ccounter = 0
		for neigh in globalNeighbors[(cid,metasim,str(h))]:
			rewhelpers = []
			try:
				rewhelpers = getRewards[(protagonist,neigh[0])]
			except KeyError:
				continue
			for rewhelper in rewhelpers:
				if (rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1] 
				and tweetid != rewhelper[7] and rewhelper[0] != rewhelper[1]):
					similar = float(neigh[1])
					if similar >= threshold:
						computeM += float(similar)
						ccounter += 1
						rews2 = []	
						try:
							rews2 = getRewards[(competitor,rewhelper[4])]
						except KeyError:
							try:
								if "_NIL" in candidates[rewhelper[4]] and rewhelper[6] < 0.5:
									inderror = []
									performance = []
									performanceConf = []
									perf = 0.0
									error = 1.0 
									confperf = 0.0	
									performance = [float(similar) * float(perf), similar]
									inderror = [float(similar) * float(error), similar]
									performanceConf = [float(similar) * float(confperf), similar]
									performanceRate.append(performance)
									confidenceAdjustedPerformanceRate.append(performanceConf)
									independentErrorRate.append(inderror)
								else:
									continue
							except:
								continue
						for rew2 in rews2:
							if rew2[0] != sid and rew2[2] == competitor and rew2[7] == rewhelper[7]: 
								inderror = []
								performance = []
								performanceConf = []
								perf = float(rew2[6])
								error = 1.0 - float(rew2[6])
								confperf = float(rewhelper[5])*float(rew2[6])		
								performance = [float(similar) * float(perf), similar]
								inderror = [float(similar) * float(error), similar]
								performanceConf = [float(similar) * float(confperf), similar]
								performanceRate.append(performance)
								confidenceAdjustedPerformanceRate.append(performanceConf)
								independentErrorRate.append(inderror)
		totalinderror = 0.0
		inderrorcounter = 0.0
		for ind in independentErrorRate:
			totalinderror += ind[0]
			inderrorcounter += ind[1]
		indresult = 0
		if not(inderrorcounter == 0):
			indresult = float(totalinderror)/float(inderrorcounter)
		totalperformance = 0.0
		performancecounter = 0.0
		for perf in performanceRate:
			totalperformance += perf[0]
			performancecounter += perf[1]
		perfresult = 0
		if not(performancecounter == 0):
			perfresult = float(totalperformance)/float(performancecounter)
		confperformance = 0.0
		confperfcounter = 0.0
		for confperf in confidenceAdjustedPerformanceRate:
			confperformance += confperf[0]
			confperfcounter += confperf[1]
		confperfresult = 0
		if not(confperfresult == 0):
			confperfresult = float(confperformance)/float(confperfcounter)
		density = float(computeM / bounds[(metasim, h)])
		if float(density) > 1.0:
			density = 1.0
		RobustnessVectors.append([[protagonist, competitor],sid, cid, metasim, 
			[[confperfresult, "ConfPerf", density], [perfresult, "Perf", density],
			 [indresult, "Error", density]],globalT])
		predictors.extend([ [indresult, density, metasim, protagonist, "IndError", competitor],
		 [perfresult, density, metasim, protagonist, "Perf", competitor], 
		 [confperfresult, density, metasim, protagonist, "ConfPerf", competitor]])
	return predictors

def annotateCandidatesInText(candidates, beginCandidates, endCandidates, subset, text):	
	counter = 0
	starts = []
	text = text.decode("utf-8")              	
	newText = text	
	for s in subset:
                cands = candidates[s].decode("utf-8")
		if not('_NIL' in cands):
			counter +=1
			if (text.count(cands) == 1) and (cands != "NIL" ):
				newText = newText.replace(cands,'<entity>' + cands + '</entity>')
			else:
				allindexes = [m.start() for m in re.finditer(candidates[s], text)]
				for ind in allindexes:
					if(str(ind) == beginCandidates[s]):
						if counter == 1:
							newText = newText[:int(beginCandidates[s])] + newText[int(beginCandidates[s]):int(endCandidates[s])].replace(cands,'<entity>' + cands + '</entity>') + newText[int(endCandidates[s]):]
						else:
							toAdd = 0
							for start in starts:
								if int(beginCandidates[s]) > int(start):
									toAdd += 17
							newText = newText[:int(beginCandidates[s])+toAdd] + newText[int(beginCandidates[s])+toAdd:int(endCandidates[s])+toAdd].replace(cands,'<entity>' + cands + '</entity>') + newText[int(endCandidates[s])+toAdd:]
			starts.append(beginCandidates[s])				
 	return newText

def initWeights(experts, metasims, dependencies, t):
	for expert in experts:
		initWeight(expert, metasims, dependencies, t)

def initWeight(expert, metasims, dependencies, t):
	global expert_weight_dic
	for ms in metasims:
		for dependency in dependencies:
			expert_weight_dic[(expert, ms, dependency, t)] = 1.0

def defaultUpdateWeight(expert, metasims, dependencies, t):
	global expert_weight_dic
	for ms in metasims:
		for dependency in dependencies:
			expert_weight_dic[(expert, ms, dependency, t)] = expert_weight_dic[(expert, ms, dependency, str(int(t)-1))] 

def dealWithWeights_pre(weighted_samples, experts, experts2, metasims, t, bounds, h, threshold, mode, 
	dependencies, chosen_experts):
	global globalmode
	globalmode = "post"
	print("Dealing w/ weights H=" +str(h))
	uniqueStates = set()
	memorizer = set()
	results = dict()
	D = dict()
	give_back = []
	for candidate_state in weighted_samples: 
		D[(candidate_state[0], candidate_state[1])] = 1.0
		for expert in experts:
			result = newLivedoM(expert, candidate_state[0], candidate_state[1], metasims, h, 
				threshold, bounds)
			try:
				current = results[(expert, candidate_state[0], candidate_state[1])]
				current.append([result, "LiveDOM"])
				results[(expert, candidate_state[0], candidate_state[1])] = current
			except:
				results[(expert, candidate_state[0], candidate_state[1])] = [[result, "LiveDOM"]]
			uniqueStates.add(candidate_state[0])
			D[candidate_state[1]] = 1.0
		for expert1 in experts:
			for expert2 in experts:
				if expert1 != expert2:
					result2 = newLivedoIPairwise(expert1, expert2, candidate_state[0], 
						candidate_state[1], metasims, h, threshold, bounds)
					try:
						current = results[(expert1, candidate_state[0], candidate_state[1])]
						current.append([result2, ("LiveDOI", expert2)])
						results[(expert1, candidate_state[0], candidate_state[1])] = current
					except:
						results[(expert1, candidate_state[0], candidate_state[1])] = [[result2, 
						("LiveDOI", expert2)]]
			for expert2 in experts2:
				result3 = newLivedoR(expert1, expert2, candidate_state[0], candidate_state[1],
				 metasims, h, threshold, bounds)
				try:
					current = results[(expert1, candidate_state[0], candidate_state[1])]
					current.append([result3, ("LiveDOR", expert2)])
					results[(expert1, candidate_state[0], candidate_state[1])] = current
				except:
					results[(expert1, candidate_state[0], candidate_state[1])] = [[result3,
					 ("LiveDOR", expert2)]]
	if h != "0":
		pd, pd2 = makePD_hedge(results, D, t, chosen_experts)
		real_pd = getRealProbs(pd)
		pds = [real_pd,pd2]
		total = sum(w for c, w in pd)
		for us in uniqueStates:
			allRews = []
			try:
				allRews = allrewards[us]
			except:
				continue
			for rew in allRews:
				if rew[0] == rew[1]:
					continue
				if mode == "batch":
					updateWeights(rew[2], total, rew[0], rew[1], rew[6], pds, metasims, t, mode,
					 dependencies, False)
				memorizer.add(rew[2])
				give_back.append([rew, results[rew[2], rew[0], rew[1]]])
		print("HAVE TO DEF UPDATE WEIGHTS!!!!!!!!")
		if mode == "batch":
			for expert in experts:
				if not(expert in memorizer):
					print("Expert " +str(expert) + " not chosen!!!")
					defaultUpdateWeight(expert, metasims, dependencies, str(int(t)+1))
	return give_back, uniqueStates, pds, results, D

def updateWeights(expert, total, state, candi, retrieved_reward, pds, metasims, t, mode, dependencies, bonus):
	global expert_weight_dic
	pd, pd2 = pds
	if float(total) == 0.0:
		total = 0.1
	global p_min
	p_min_global = 0.1
	p_expert = 0.0
	assigned = 0
	for p in pd:
		if p[0][0] == expert and p[0][1] == state:
			p_expert += (float(p[1]) / float(total))
			assigned += 1
	for ms in metasims:
		for dependency in dependencies:
			prediction = -1
			density = -1
			found = 0
			for p in pd2:
				if p[0][0] == expert and p[3] == ms and p[0][1] == state and p[0][2] == candi and dependency == p[4]:
					prediction, density = p[2]
					found = 1
					break
			if found == 0:
				try:
					updated_weight = expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
				except:
					expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = expert_weight_dic[(expert, ms, dependency, str(int(t)))]
			else:
				reward = ( 1 - abs( float(retrieved_reward) - float(prediction)  ))			
				reward = mapIntervals(reward)
				reward = float(reward) * float(density)
				p_expert += p_min_global
				reward = float(reward) / float(p_expert) 
				try:
					updated_weight = expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
				except:
					expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = expert_weight_dic[(expert, ms, dependency, str(int(t)))]
				if mode == "batch":
					expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = expert_weight_dic[(expert, ms, dependency, str(t))]
				else:
					try:	
						update = math.exp(reward * p_min_global)
						expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = float(expert_weight_dic[(expert, ms, dependency, str(t))]) * float(math.exp(reward * p_min_global))
					except:	
						update = 0.0
					 	try:
							update = math.exp(reward * p_min)
							expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = expert_weight_dic[(expert, ms, dependency, str(t))]						
						except:
							expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = math.exp(100)

def getCRs():
	global candidateRewards
	for rew in rewards:
		candidateRewards[rew[4]] = rew[6]

def dealWithWeights_post(uniqueStates, pds, t, metasims, mode, dependencies, experts, winna,
 chosen_experts):
	memorizer = set()
	pd, pd2 = pds
	total = sum(w for c, w in pd)
	if mode == "train":
		for us in uniqueStates:
			collector = dict()
			allRews = []
			try:
				allRews = allrewards[us]
			except:
				continue
			for rew in allRews:
				bonus = False
				if rew[0] == rew[1]:
					continue
				if (not(rew[4]) in winna and candidateRewards[rew[4]] >= 0.5) or (rew[4] in 
					winna and candidateRewards[rew[4]] < 0.5):
					bonus = True
				print("Updated weights for " +str(rew[2]))
				updateWeights(rew[2], total, rew[0], rew[1], rew[6], pds, metasims, t, mode,
				 dependencies, bonus)
				memorizer.add(rew[2])
				try:
					collector[rew[2]].append(rew)
				except:
					collector[rew[2]] = []
					collector[rew[2]].append(rew)
			if len(collector) > 0:
				for expert in collector:
					rews = collector[expert] 
					pairs = datapoints[rew[7]][3]
					count = 0
					for rew in rews:
						count += rew[6]
					diff = len(pairs) - rew[6]
	else:
		for us in uniqueStates:
			allRews = []
			try:
				allRews = allrewards[us]
			except:
				continue
			for rew in allRews:
				bonus = False
				if rew[0] == rew[1]:
					continue
				assessWeights(rew[2], total, rew[0], rew[1], rew[6], pd2, metasims, t,
				 mode, dependencies, bonus)
	for expert in experts:
		if not(expert in memorizer):
			defaultUpdateWeight(expert, metasims, dependencies, str(int(t)+1))

def apriori_assess(experts, experts2, weighted_samples, metasims, h, bounds):
	global globalmode
	globalmode = "pre"
	pds = []
	results = dict()
	for sample in weighted_samples:
		for candidate in sample:
			for expert in experts:
				result = newLivedoM(expert, candidate[0], candidate[1], metasims, h, 0.2, bounds)
				try:
					current = results[(expert, candidate[0], candidate[1])]
					current.append([result, "LiveDOM"])
					results[(expert, candidate[0], candidate[1])] = current
				except:
					results[(expert, candidate[0], candidate[1])] = [[result, "LiveDOM"]]
		#		print(">->->->->->->>>> " +str(result))
			for expert1 in experts:
				for expert2 in experts:
					if expert1 != expert2:
						result2 = newLivedoIPairwise(expert1, expert2, candidate[0], candidate[1],
						 metasims, h, 0.2, bounds)
						try:
							current = results[(expert1, candidate[0], candidate[1])]
							current.append([result2, ("LiveDOI", expert2)])
							results[(expert1, candidate[0], candidate[1])] = current
						except:
							results[(expert1, candidate[0], candidate[1])] = [[result2,
							 ("LiveDOI", expert2)]]
			for expert1 in experts:
				for expert2 in experts2:
					result3 = newLivedoR(expert1, expert2, candidate[0], candidate[1],
					 metasims, h, 0.2, bounds)
					try:
						current = results[(expert1, candidate[0], candidate[1])]
						current.append([result3, ("LiveDOR", expert2)])
						results[(expert1, candidate[0], candidate[1])] = current
					except:
						results[(expert1, candidate[0], candidate[1])] = [[result3, ("LiveDOR",
						 expert2)]]					
	return results

def choose_and_execute(budget, weights, h, t, metasims):
	global D_used
	D_used = []
	D = dict()
	chosen_experts = []
	for w in weights:
		D[w[2]] = 1.0
	all_options = []
	candis = set()
	newPD = []
	asked = []
	for i in range(budget):
		if len(weights) > 0:
			pd, pd2 = makePD_hedge(weights, D, t, chosen_experts)
			chosen = weighted_choice_hedge(pd)
			real_probs = getRealProbs(pd)
			D_used.append([D, chosen])
			temp_candis = execute(chosen[0], h)
			chosen_experts.append(chosen[0][0])
			candis.update(temp_candis)
			new_weights = dict()
			potential = dict()
			for w in weights:
				if not(w[0] == chosen[0][0] and w[1] == chosen[0][1]):
					new_weights[w] = weights[w]
			for prob in real_probs:
				if prob[0][0] == chosen[0][0] and prob[0][1] == chosen[0][1]:
					potential[prob[0][2]] = prob[1]   
			weights = new_weights
			for d in D: 
				try:
					weight = (float(potential[d]) + 1.0) / 2.0
					eps = 1.0 - float(weight)
					a = 0.5 * math.log((1.0-float(eps))/float(eps))
					D[d] = D[d] * math.exp(- a )
				except:
					pass	
	return candis, chosen_experts

def getRealProbs(choices):
	force_positives = []
	for choice in choices:
		c,w = choice
		if float(w) < 0.0:
			new_w = -1.0 * float(w)
			force_positives.append([c,new_w])
		else:
			force_positives.append([c,w])
	total = sum(w for c, w in force_positives)
	if float(total) == 0.0:
		total = 1.0
	to_raise = []
	to_stay = []
	for choice in choices:
		c, w = choice
		prob = float(w) / float(total)	
		prob = mapIntervalsBack(prob)
		if float(prob) < 0.08:
			new_weight = 0.08 
			new_choice = [c, new_weight]
			to_stay.append(new_choice) 
		else:
			to_stay.append([c, prob])
	return to_stay

def weighted_choice_hedge(choices):
	force_positives = []
	for choice in choices:
		c,w = choice
		if float(w) < 0.0:
			new_w = -1.0 * float(w)
			force_positives.append([c,new_w])
		else:
			force_positives.append([c,w])
	total = sum(w for c, w in force_positives)
	if float(total) == 0.0:
		total = 1.0
	to_raise = []
	to_stay = []
	for choice in choices:
		c, w = choice
		prob = float(w) / float(total)	
		prob = mapIntervalsBack(prob)
		if float(prob) < 0.08:
			new_weight = 0.08 
			new_choice = [c, new_weight]
			to_stay.append(new_choice) 
		else:
			to_stay.append([c, prob])
	new_total = sum(w for c, w in to_stay)
	r = random.uniform(0, new_total)
	upto = 0
	random.shuffle(to_stay)
	for c, w in to_stay:
		if upto + w >= r:
			return c, w
		upto += w
  	assert False, "Shouldn't get here"

def mapIntervals(val):
	A = 0
	B = 1
	a = -1
	b = 1
	return float((val - A)*(b-a)/(B-A) + a)

def mapIntervalsBack(val):
	A = -1
	B = 1
	a = 0
	b = 1
	return float((val - A)*(b-a)/(B-A) + a)

def makePD_hedge(results, D, t, chosen_experts):
	print("CURRENT T=" + str(t))
	global p_min
	pd_arr = []
	pd_arr_2 = []
	avg_denom = 0.0
	for expert in results:
		avg = 0.0
		for res_component in results[expert]:
			for res in res_component[0]:
				if ((res[2] == "StateTextExtraChars") or (res[2] == "StateTextLengthMetaSimilarity") 
				or (res[2] == "StateSameLingualType") or (res[2] == "StateCandidateWordEmbeddings")):
					continue
				dependency = None
				if res_component[1] == "LiveDOM":
					dependency = "LiveDOM"
					expert_1 = res[3]
					prediction = res[0]
					density = res[1]
					pd_arr_2.append([expert, prediction, [prediction, density], res[2], dependency])
				if res_component[1][0] == "LiveDOI":
					continue
				if res_component[1][0] == "LiveDOR":
					dependency = ("LiveDOR", res[4], res[5])
					prediction = res[0]
					density = res[1]
					competitor = res[5]
					behavior = res[4]	
					if not(behavior == "Perf"):
						continue
					pd_arr_2.append([expert, prediction, [prediction, density], res[2], dependency])
				if dependency == None:
					print("HA???? " +str(res_component[1]))
				prediction = mapIntervals(res[0])
				avg += float(prediction) * float(res[1]) * float(expert_weight_dic[(res[3], res[2], 
					dependency, t)]) * D[(expert[2])]
		pd_arr.append([expert, avg])
	return pd_arr, pd_arr_2

def getFromFile(algo, text):
	res = None
	text = text.replace("/","")
	if len(text) > 200:
		text = text[0:200]
	try:
		with open('./stash/execs_ ' +str(algo) + '_' + text + '.txt', 'r') as f:
			res = cPickle.load(f)
		f.close()
		return res
	except:
		return None

def save2File(algo, text, result):
	text = text.replace("/","")
	if len(text) > 200:
		text = text[0:200]
	with open('./stash/execs_ ' +str(algo) + '_' + text + '.txt', 'w') as f:
		cPickle.dump(result, f)
		f.close()

def getNewSamples(metasims, h, samplesize, threshold, currentTweetID, t, bounds, results, mode):
	global state_dic
	global Y_Label_count
	global Y_Labels
	global Y_Preds
	global Y_Label
	global Y_Label_global
	global Y_Pred
	global Y_Pred_global
	global Y_Pred_count
	global Y_Pred_1
	global Y_Label_1
	global Y_Label_count_1
	global Y_Label_1
	global Y_Pred_count_1
	global Y_Pred_Singles
	global Y_Label_Singles
	global Y_Pred_Singles_1
	global Y_Label_Singles_1
	global Y_theo
	global Y_Label_text
	global Y_Label_text_1
	global chosen_exps
	getCRs()
	reward_belief_collector_KL = [] 
	some_collector = dict()
	reward_collector = dict()
	uniqueStates = set()
	by_exp = dict()
	for result in results:
		rew = result[0]
		regressors = result[1]
		print("(" +str(rew[0]) + ") Thinking about: " +str(candidates[rew[1]]) 
			+ " saying: " +str(candidates[rew[4]]) +" by: " +str(rew[2]))
		if not("_NIL" in candidates[rew[4]]):
			Y_theo[str(h)].add((rew[4], candidates[rew[4]], rew[6]))
		try:
			by_exp[rew[4]].add(rew[2])
			prior0 = some_collector[rew[4]]
			prior0.add(rew[1])
			some_collector[rew[4]] = prior0
			prior = reward_collector[rew[4]]
			prior.append(regressors)
			reward_collector[rew[4]] = prior
		except:
			by_exp[rew[4]] = set()
			by_exp[rew[4]].add(rew[2])
			some_collector[rew[4]] = set()
			some_collector[rew[4]].add(rew[1])
			prior = []
			prior.append(regressors)
			reward_collector[rew[4]] = prior
	for result in results:
		rew = result[0]
		regressors = result[1]
	turned_collector = dict()
	final_collector = dict()	
	denom = 0.0	
	all_best = []
	for best in reward_collector:
		all_best.extend(reward_collector[best])
	denom = getDenom(all_best, t)
	for best in reward_collector:
		rew = computeItSimple(reward_collector[best], t, 0.0, "not_test", denom)
		turned_collector[best] = rew
	for best in turned_collector:
		rew = turned_collector[best]
	to_alter = [] 
	normalizer = 0.0
	for best1 in turned_collector:
		final_collector[best1] = turned_collector[best1]
		normalizer += float(final_collector[best1])
	clusters = []
	for best in final_collector:
		p = best
		r2 = range(int(beginCandidates[p]), int(endCandidates[p]))
		r2 = set(r2)
		rew = final_collector[best]
		overlap = 0
		for cluster in clusters:
			for r in cluster:
				if len(r[0].intersection(r2)) > 0:
					if not [r2,p,rew] in cluster:
						cluster.append([r2,p,rew])
					overlap = 1
		if overlap == 0:
			clusters.append([[r2,p,rew]])	
	possies = dict()						
	choices = []
	weighted_choices = []
	iterations = range(40)
	for iteration in iterations:	
		choice = []	
		for cluster in clusters:
			begin = 0
			end = 0
			tochoose = []
			todraw =  []
			for part in cluster:
				nila = 0
				rews = []
				if type(part[0]) is list:
					if type(part[0][0]) is set:
						for el in part:
							part_rew = float(el[2])
							possies[el[1]] = el[1]
							rews.append([el[1], part_rew]) 
						tochoose.append(weighted_choice_hedge(rews))						
				else:
					if type(part[0]) is set:
						part_rew = float(part[2])
						if begin > int(beginCandidates[part[1]]):
							begin = int(beginCandidates[part[1]])
						if end < int(endCandidates[part[1]]):
							end = int(endCandidates[part[1]])						
						possies[part[1]] = part[1]
						todraw.append([part[1], part_rew])
			todraw.extend(tochoose)
			iters = range(40)
			foundthebitch = 1
			for iteri in iters:
				proposition = weighted_choice_hedge(todraw)	
				for prior in choice:
					range1 = []
					range2 = []
					try:
						range1 = range(int(beginCandidates[prior[0]]), 
							int(endCandidates[prior[0]])+1)
					except KeyError:
						continue
					try:
						range2 = range(int(beginCandidates[proposition[0]]), 
							int(endCandidates[proposition[0]])+1)
					except KeyError:
						continue
					overlap = len(set(range1).intersection(set(range2)))
					if prior[0] == proposition[0] or overlap > 0:
						foundthebitch = 0
						break
				if foundthebitch == 1:
					choice.append(proposition)
					break
		skinnychoice = []
		print("CHOICE")
		weighted_choices.append(choice)
		for ch in choice:
			if not '_NIL' in ch[0]:
				skinnychoice.append(ch[0])
			try:
				print("-> " +candidates[ch[0]] + ") " +str(by_exp[ch[0]]))
			except KeyError:
				print("-> THAT NIL THING")
		choices.append(skinnychoice)
	combiIDcounter = 0
	regrets = []
	leader = 0
	current = 0.0
	counti = -1
	for cluster in choices:	
		counti += 1	
		totalreg = 0.0
		for part in cluster:		
			totalreg += float(final_collector[part])
		regrets.append(totalreg)
		if totalreg > current:
			current = totalreg
			leader = counti
#	mychoice = choices[leader]
	mychoice = random.sample(choices,1)
	mychoice = mychoice[0]
	if mode == "test":
		print("I chose ... ")
		for possi in possies:
			flag = -1
			for ch in mychoice:
				if possi == ch:
					flag = 1
			if flag == -1:
				Y_Pred_global.append(0)
			else:
				Y_Pred_global.append(1)
			res = round(candidateRewards[possi])
			Y_Label_global.append(res)
		if h == "1":
			Y_Label_count += len(datapoints[currentTweetID][3])
			Y_Labels.extend(datapoints[tweetid[currentTweetID]][3])
			for part in mychoice:
				try:
					print("-> " +candidates[part] + " (" +str(candidateRewards[part])
					 + ") " +str(by_exp[part]))
				except:
					pass
				if not("_NIL" in candidates[part]):
					Y_Pred.append(1.0)
					Y_Label.append(candidateRewards[part])
					Y_Label_text.append(candidates[part])
					Y_Pred_count += int(candidateRewards[part])
					Y_Preds.append(candidates[part])
		else:
			Y_Label_count_1 += len(datapoints[currentTweetID][3])
			for part in mychoice:
				print("-> " +candidates[part] + " (" +str(candidateRewards[part]) 
					+ ")")
				try:
					chosen_exps.append(by_exp[part])
				except:
					chosen_exps.append("Add_NIL")
				if not("_NIL" in candidates[part]):
					Y_Pred_1.append(1.0) #TODO just use "1"?!
					Y_Label_1.append(candidateRewards[part])
					Y_Label_text_1.append(candidates[part])
					Y_Pred_count_1 += int(candidateRewards[part])
	final_combis = []
	combiIDcounter = 0	
	state_choices = []
	for combi in [mychoice]:
		if len(combi) == 0:
			continue
		newStateID = currentTweetID + str(combiIDcounter)
		newText = annotateCandidatesInText(candidates, beginCandidates, endCandidates, 
			combi, stateText[currentTweetID])		
		textWithEntites[newStateID] = newText
		newState = []
		for p in combi:
			candidateHorizon[p] = "1"
			if not("_NIL" in candidates[p]):
				newState.append([newStateID, p])
		state_choices.append(newState)
		state_dic[newStateID] = combi
		stateText[newStateID] = stateText[currentTweetID]
		tweetid[newStateID] = currentTweetID
		combiIDcounter += 1
	return state_choices, mychoice	

def getDenom(results, t):
	global p_min
	global cmon
	preds = []
	ap = 0.0
	count = 0
	regressor = []
	avg_denom = 0.0
	for result1 in results:
		avg = 0.0
		for result in result1:
			global_dependency = result[1]
			for res in result[0]:
				print("RES " +str(res))
				helpi = ""
				dependency = None
				if global_dependency == "LiveDOM":
					dependency = "LiveDOM"
				if global_dependency[0] == "LiveDOI":
					dependency = ("LiveDOI", res[4], res[5])
					helpi = res[4]
					continue
				if global_dependency[0] == "LiveDOR":
					dependency = ("LiveDOR", res[4], res[5])
					expert = res[3]
					prediction = res[0]
					density = res[1]
					competitor = res[5]
					behavior = res[4]	
					if not(behavior == "IndError"):
						continue
				if dependency == None:
					print("WHY??? " +str(global_dependency))
				avg_denom += float(res[0]) * float(expert_weight_dic[(res[3], res[2], dependency, t)])
	return avg_denom

def computeItSimple(results, t, label, mode, denom):
	avg = 0.0
	doi_factor = dict()
	doi_factor_perf = dict()
	all_exps = set() 
	for result1 in results:
		for result in result1:
			global_dependency = result[1]
			for res in result[0]:
				expert = res[3]
				all_exps.add(expert)
				if global_dependency[0] == "LiveDOI":
					prediction = res[0]
					density = res[1]
					competitor = res[5]
					behavior = res[4]
					if behavior == "IndError":
						factor = float(prediction) * float(density)
						try:
							doi_factor[expert].append([competitor, res[2], factor])
						except:
							doi_factor[expert] = []
							doi_factor[expert].append([competitor, res[2], factor])
					else:
						if behavior == "ConfPerf":
							factor = float(prediction) * float(density)
							try:
								doi_factor_perf[expert].append([competitor, res[2], factor])
							except:
								doi_factor_perf[expert] = []
								doi_factor_perf[expert].append([competitor, res[2], factor])
	for result1 in results:
		for result in result1:
			global_dependency = result[1]
			for res in result[0]:
				this_expert = res[3]
				is_doi = False
				if ((res[2] == "StateTextExtraChars") or (res[2] == "StateTextLengthMetaSimilarity")
				 or (res[2] == "StateSameLingualType") or (res[2] == "StateCandidateWordEmbeddings")):
					continue
				helpi = ""
				dependency = None
				if global_dependency == "LiveDOM":
					dependency = "LiveDOM"
				if global_dependency[0] == "LiveDOI":
					continue
					is_doi = True
				if global_dependency[0] == "LiveDOR":
					dependency = ("LiveDOR", res[4], res[5])
					prediction = res[0]
					density = res[1]
					competitor = res[5]
					behavior = res[4]	
					if not(behavior == "Perf"):
						continue
				if dependency == None:
					print("WHY??? " +str(global_dependency))
				doi_impact = 1.0
				inds = 0.0
				all_doi_factors = doi_factor[this_expert]
				for fac in all_doi_factors:
					if fac[0] in all_exps and fac[1] == res[2]:
						pred = fac[2]
						inds += (1.0 - float(pred))
				doi_impact = 1.0 / (float(inds) + 1.0)	
				inds_perf = 0.0
				doi_impact_perf = 1.0
				flag = False
				all_doi_factors_perf = doi_factor_perf[this_expert]
				for fac in all_doi_factors_perf:
					if fac[0] in all_exps and fac[1] == res[2]:
						flag = True
						pred = fac[2]
						inds_perf += float(pred)
				if flag:
					doi_impact_perf = float(inds_perf)  /  ( 1.0 + float(inds_perf) )	
				mapped = mapIntervals(float(res[0]))
				dor_impact = 1.0
				if global_dependency[0] == "LiveDOR":
					dor_impact = 0.5
				avg += (mapped * float(res[1]) * float(doi_impact_perf) * float(doi_impact) 
				* float(dor_impact) * float(expert_weight_dic[(res[3], res[2], dependency, t)]))			
	return avg

def findID(sol, begin, end):
	uptop = 3
	rangePlus = range(uptop)
	rangeMinus = []
	for ra in rangePlus:
		b = -1 * ra
		rangeMinus.append(b)
	completeRange = rangeMinus + rangePlus
	reducedRange = set(completeRange)	
	for ra in reducedRange:
		newBegin = str(int(begin)+ra)
		newEnd = str(int(end)+ra)	
		candid = sol + newBegin + newEnd	
		candid0 = "0000" + sol + newBegin + newEnd
		try:
			found = candidates[candid0,candid]
			return candid
		except:
			pass
	for ra in reducedRange:
		newBegin = begin
		newEnd = str(int(end)+ra)	
		candid = sol + newBegin + newEnd	
		candid0 = "0000" + sol + newBegin + newEnd
		try:
			found = candidates[candid0,candid]
			return candid
		except:
			pass
	for ra in reducedRange:
		newBegin = str(int(begin)+ra)
		newEnd = end	
		candid = sol + newBegin + newEnd	
		candid0 = "0000" + sol + newBegin + newEnd
		try:
			found = candidates[candid0,candid]
			return candid
		except:
			pass
	candid = sol + begin + end	
	candid0 = "0000" + sol + begin + end
	return candid0, candid	


def getWholeWord(tweet, ne):
	ww = None
	possibilities = ["", "#", ",", "."]
	for po in possibilities:
		ww = getWholeW(tweet, ne, po)
		if len(ww) > 0:
			break
	if len(ww) == 0:
		print("FUCK, nothing for " +str(ne) + "IN " +str(tweet))
	return ww

def getWholeW(tweet, ne, to_remove):
	temp = ngrams(tweet.split(), len(ne.split()))
	k = 0
	results = []
	for t in temp:
		newString = ""
		for i in range(len(ne.split())):
			option = t[i]
			option = option.replace(to_remove,"")
			if(i != 0):
				newString = newString + " " + option
			else:
				newString = option
		if ne in newString:
			found = tweet[k:].find(newString)	
			result = []
			result.append(newString)
			k += found
			result.append(k)
			k += len(newString)
			result.append(k)
			results.append(result)
			k += 1
	return results

def getFeatureID(feature):
	if feature == "CoherenceEntityLinks":
		return "1010101011"
	if feature == "MentionPrior":
		return "1010101012"
	if feature == "MentionInfoSourceWithRepresentativeSentences":
		return "1010101013"
	if feature == "SyntacticBasedContextualEntityFrequencyOccurence":
		return "1010101014"
	if feature == "StanfordNEROutput":
		return "1010101015"
	if feature == "IllinoisNETOutput":
		return "1010101016"
	if feature == "BalieOutput":
		return "1010101017"
	if feature == "OpenNLPOutput":
		return "1010101018"
	if feature == "WordPhrases":
		return "1010101019"
	if feature == "SyntacticTextFeature":
		return "1010101020"

def executeAGDISTIS(algoid, aTweet):
	parameterizedFeatures = []
	parameterizedFeatures.append(["CoherenceEntityLinks",
	 algoid+getFeatureID("CoherenceEntityLinks"), "1"])
	parameterizedFeatures.append(["MentionPrior",
	 algoid+getFeatureID("MentionPrior"), "1"])
	ensemble = []
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("AGDISTIS")
	algoInfo.append("Entity Linking")
	algoInfo.append("Mention-Entity Graph")
	algoInfo.append("HITS")
	algoInfo.append("DBpedia")
	algoInfo.append(ensemble)
	algoInfo.append("NoEnsemble")
	results = []
	data = {'text':'%s' % aTweet, 'type': 'agdistis'}
	r = requests.post('http://139.18.2.164:8080/AGDISTIS', data)
	res0 = r.text
	toShow = aTweet.encode('utf-8')
	tempres = res0.encode('utf-8')
#	print("Tweet: " +str(toShow) + ", RES: " +str(tempres))
	res = json.loads(res0)
	for r in res:
		lenText = len(r['namedEntity'])
		added = lenText + r['start']
		result = []
		result.append(algoInfo)
		result.append(str(r['namedEntity']))
		disamb = ""
		if r['disambiguatedURL'] is None:
			continue
#			disamb = str(r['namedEntity']) + "_NIL"
		else:
			disamb = r['disambiguatedURL']
		result.append(disamb)
		result.append(str(r['start']))
		result.append(str(added))
		result.append(str(0.5))
		results.append(result)	
	if len(results) == 0:
		result = []
		result.append(algoInfo)
		results.append(result)
	return results 	


def executeAIDATagger(algoid, aTweet):
	parameterizedFeatures = []
	parameterizedFeatures.append(["CoherenceEntityLinks",
	 algoid+getFeatureID("CoherenceEntityLinks"), "1"])
	parameterizedFeatures.append(["MentionPrior", 
		algoid+getFeatureID("MentionPrior"), "1"])
	parameterizedFeatures.append(["SyntacticBasedContextualEntityFrequencyOccurence",
	 algoid+getFeatureID("SyntacticBasedContextualEntityFrequencyOccurence"), "1"])
	parameterizedFeatures.append(["MentionInfoSourceWithRepresentativeSentences",
	 algoid+getFeatureID("MentionInfoSourceWithRepresentativeSentences"), "1"])
	ensemble = []
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("AIDA")
	algoInfo.append("Entity Linking")
	algoInfo.append("Mention-Entity Graph")
	algoInfo.append("MST")
	algoInfo.append("YAGO")
	algoInfo.append(ensemble)
	algoInfo.append("NoEnsemble")
	aTweet = aTweet.replace('<entity>','[[')
	aTweet = aTweet.replace('</entity>',']]')	
	results = []
	data = {'text':'%s' % aTweet, 'tag_mode': 'manual'}
	r = requests.post('https://gate.d5.mpi-inf.mpg.de/aida/service/disambiguate',
	 data)
	res0 = r.text
#	print(res0)
	res = json.loads(res0)
	for r in res['mentions']:
		if len(r['allEntities']) > 0:	
			result = []
			result.append(algoInfo)
			result.append(r['name'])
			result.append(r['allEntities'][0]['kbIdentifier'])
			result.append(str(r['offset']))
			result.append(str(r['offset'] + r['length']))
			result.append(str(r['allEntities'][0]['disambiguationScore']))
			results.append(result)		
	if len(results) == 0:
		result = []
		result.append(algoInfo)
		results.append(result)	
	return results 

def executeFOX(algoid, tweet, ensembleids):
	parameterizedFeatures = []
	parameterizedFeatures.append(["StanfordNEROutput", algoid+getFeatureID("StanfordNEROutput"), "1"])
	parameterizedFeatures.append(["IllinoisNETOutput", algoid+getFeatureID("IllinoisNETOutput"), "1"])
	parameterizedFeatures.append(["BalieOutput", algoid+getFeatureID("BalieOutput"), "1"])
	parameterizedFeatures.append(["OpenNLPOutput", algoid+getFeatureID("OpenNLPOutput"), "1"])
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("FOX")
	algoInfo.append("Entity Recognition")
	algoInfo.append("(Non-) Linear Combination of NER with MLP")
	algoInfo.append("Whatever")
	algoInfo.append("Training Sets of Base Learners")
        algoInfo.append(ensembleids)
	algoInfo.append("Ensemble")
	results = []
	data = {'input':'%s' % tweet, 'type': 'text', 'task': 'ner', 'output': 'Turtle', 'foxlight': 'OFF', 'disamb': 'off'}
	headers = {'Content-Type': 'application/json'}
#	r = requests.post('http://aifb-ls3-romulus.aifb.kit.edu:4444/call/ner/entities', data=json.dumps(data), headers=headers)
	r = requests.post('http://fox-demo.aksw.org/call/ner/entities', data=json.dumps(data), headers=headers)
	result = r.text
	#print(result)
	g = Graph()
	g.parse(data=result, format="n3")
	qres = g.query(
    		"""SELECT ?cand ?start ?end
       		WHERE {
        	_:blankNode <http://www.w3.org/2000/10/annotation-ns#body> ?cand;
        	   <http://ns.aksw.org/scms/beginIndex> ?start;
        	   <http://ns.aksw.org/scms/endIndex> ?end .
       		}""")
	for row in qres:											
		result = []
		result.append(algoInfo)
		result.append(str(row["cand"]))
		result.append(str(row["start"]))
		result.append(str(row["end"]))
		result.append(0.5)
		results.append(result)
	finalres = []
	for result in results:
		boole = 1
		if result[2] > result[3]:
			boole = 0
		else:
			end = int(result[2]) + len(result[1])
			if(end != int(result[3])):
				boole = 0
		if boole == 1:
			finalres.append(result)
	if len(finalres) == 0:
		result = []
		result.append(algoInfo)
		finalres.append(result)			
	return finalres		


def executeStanfordNER(algoid, tweet):
	parameterizedFeatures = []
	parameterizedFeatures.append(["SyntacticTextFeature", algoid+getFeatureID("SyntacticTextFeature"), "1"])
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("Stanford NER")
	algoInfo.append("Entity Recognition")
	algoInfo.append("Conditional Random Field Local-Only Model")
	algoInfo.append("Viterbi Inference")
	algoInfo.append("CoNLL, MUC-6, MUC-7, ACE")
        algoInfo.append([])
	algoInfo.append("NoEnsemble")
	results = []
	data = {'input':'%s' % tweet, 'type': 'text', 'task': 'ner', 'output': 'Turtle', 'foxlight': 'org.aksw.fox.tools.ner.en.NERStanford', 'disamb': 'off'}
	headers = {'Content-Type': 'application/json'}
	r = requests.post('http://fox-demo.aksw.org/call/ner/entities', data=json.dumps(data), headers=headers)
#	r = requests.post('http://aifb-ls3-romulus.aifb.kit.edu:4444/call/ner/entities', data=json.dumps(data), headers=headers)
	result = r.text
	g = Graph()
	g.parse(data=result, format="n3")
	qres = g.query(
    		"""SELECT ?cand ?start ?end
       		WHERE {
        	_:blankNode <http://www.w3.org/2000/10/annotation-ns#body> ?cand;
        	   <http://ns.aksw.org/scms/beginIndex> ?start;
        	   <http://ns.aksw.org/scms/endIndex> ?end .
       		}""")
	for row in qres:											
		result = []
		result.append(algoInfo)
		result.append(str(row["cand"]))
		result.append(str(row["start"]))
		result.append(str(row["end"]))
		result.append(0.5)
		results.append(result)
	finalres = []
	for result in results:
		boole = 1
		if result[2] > result[3]:
			boole = 0
		else:
			end = int(result[2]) + len(result[1])
			if(end != int(result[3])):
				boole = 0
		if boole == 1:
			finalres.append(result)	
	if len(finalres) == 0:
		result = []
		result.append(algoInfo)
		finalres.append(result)			
	return finalres					

def executeSpotlightTagger(algoid, tweet, combis, candidates, beginCandidates, 
	confidence, support):
	tweet = tweet.replace("&", "&amp;")
	tweet = tweet.replace("\"", "&quot;" )
	parameterizedFeatures = []
	parameterizedFeatures.append(["WordPhrases", 
		algoid+getFeatureID("WordPhrases"), "1"])
	ensemble = []
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("Spotlight Tagger")
	algoInfo.append("Entity Linking")
	algoInfo.append("Dictionairy Lookup")
	algoInfo.append("Brute Force Search")
	algoInfo.append("????")	
	algoInfo.append(ensemble)
	algoInfo.append("NoEnsemble")
	tweet = tweet.replace('<entity>','')
	tweet = tweet.replace('</entity>','')
	tweet = ('<?xml version="1.0" encoding="UTF-8"?> <annotation text="'
	 + tweet + '">')
	for cid in combis:
		enc = candidates[cid].replace("&", "&amp;")
		if not('_NIL' in enc):
			tweet = (tweet + ' <surfaceForm name="{0}" offset="{1}" /> '
			.format(enc, int(beginCandidates[cid])))
	tweet = tweet + '</annotation>'
	results = []
	data = {'text': '%s' % tweet, 'confidence': '%i' % confidence, 
	'support': '%i' % support}
	headers = {'accept': 'application/json'}
	r = requests.post('http://aifb-ls3-remus.aifb.kit.edu:2225/rest/disambiguate', 
		data=data, headers=headers)
	res = r.json()
	try:
		test = res['Resources']
	except KeyError:
		return results
	for i in res['Resources']:
		lenText = len(i['@surfaceForm'])
		offset = i['@offset']
		added = lenText+int(offset)
		result = []
		result.append(algoInfo)
		result.append(i['@surfaceForm'])
		result.append(i['@URI'])
		result.append(str(i['@offset']))
		result.append(str(added))
		result.append(confidence)
		result.append(str(i['@support']))
		result.append(str(i['@similarityScore']))
		result.append(str(i['@percentageOfSecondRank']))
		results.append(result)	
	if len(results) == 0:
		result = []
		result.append(algoInfo)
		results.append(result)
	return results


def executeSpotlightSpotter2(algoid, tweet):
	parameterizedFeatures = []
	parameterizedFeatures.append(["WordPhrases", 
		algoid+getFeatureID("WordPhrases"), "1"])
	ensemble = []
	algoInfo = []
	algoInfo.append(algoid)
        algoInfo.append("")
	algoInfo.append(parameterizedFeatures)
	algoInfo.append("Spotlight Spotter")
	algoInfo.append("Entity Recognition")
	algoInfo.append("Dictionairy Lookup")
	algoInfo.append("Brute Force Search")
	algoInfo.append("????")
	algoInfo.append(ensemble)
	algoInfo.append("NoEnsemble")
	results = []
	results = []
	data = {'text': '%s' % tweet, 'confidence': '%i' % 0.2}
	headers = {'accept': 'application/json'}
	r = requests.post('http://aifb-ls3-remus.aifb.kit.edu:2225/rest/annotate',
	 data=data, headers=headers)
	res = r.json()
	try:
		test = res['Resources']
	except KeyError:
		return results
	for i in res['Resources']:
		lenText = len(i['@surfaceForm'])
		offset = i['@offset']
		added = lenText+int(offset)
		result = []
		result.append(algoInfo)
		result.append(i['@surfaceForm'])
		result.append(str(i['@offset']))
		result.append(str(added))
		result.append(0.5)
		results.append(result)	
	if len(results) == 0:
		result = []
		result.append(algoInfo)
		results.append(result)
	return results

def execute(chosen, h):
	global explore_mode
	candis = set()
	candis_1 = set()
	global stateCandis
	global rewards
	global allrewards
	global candidateHorizon
	expert = chosen[0]
	sid = chosen[1]
	tweet = stateText[sid]
	pairs = datapoints[tweetid[sid]][3]
	resultset = []
	if expert == "01":
		resultset = getFromFile("01",tweet)
		if resultset == None:
			resultset = executeFOX("01", tweet, ["04"])
			save2File("01", tweet, resultset)
	if expert == "02":
		resultset = getFromFile("02",tweet)
		if resultset == None:
			resultset = executeSpotlightSpotter2("02", tweet)
			save2File("02", tweet, resultset)
	if expert == "03":
		resultset = getFromFile("03",tweet)
		if resultset == None:
			resultset = executeStanfordNER("03", tweet)
			save2File("03", tweet, resultset)
	if expert == "11":
		if "<entity>" in textWithEntites[sid]:
			resultset = getFromFile("11",textWithEntites[sid])		
			if resultset == None:
				resultset = executeAGDISTIS('11', textWithEntites[sid])
				save2File("11", textWithEntites[sid], resultset)
	if expert == "12":
		if "<entity>" in textWithEntites[sid]:
			resultset = getFromFile("12",textWithEntites[sid])
			if resultset == None:
				resultset = executeAIDATagger('12', textWithEntites[sid])
				save2File("12", textWithEntites[sid], resultset)
	if expert == "13":
		if "<entity>" in textWithEntites[sid]:
			resultset = getFromFile("13",textWithEntites[sid])
			if resultset == None:
				resultset = executeSpotlightTagger('13', textWithEntites[sid],
				 state_dic[sid], candidates, beginCandidates, 0.3, 0)
				save2File("13", textWithEntites[sid], resultset)
	try:
		prior = executed[sid]
		prior.append(expert)
		executed[sid] = prior
	except:
		prior = [expert]
		executed[sid] = prior
	tempcandidates = []
	tempcandidatesWONIL = []
	localresults = []
	something = 0
	sol = sid
	state = sid
	retrieved = 0
	new_resultset = []
	for result in resultset:
			if not(len(result) == 1):
				new_resultset.append(result)
	for pair in pairs:
		tweet = tweet.decode("utf-8")
		found = False
		for result in resultset:
			if not(len(result) == 1):
				ne1 = result[1].decode("utf-8")
				if (len(get_overlap(ne1,pair[0])) >= (len(pair[0]) - 1)
				 and len(get_overlap(ne1,pair[0]))) >= 1:
					found = True
					break
		if not(found) and int(expert) < 9:
			ww = getWholeWord(tweet, pair[0])
			initial_word = None
			begin = -1
			end = -1
			try:
				bla = ww[0][0]
			except:
				pass
			initial_word = ww[0][0]
			begin = ww[0][1]
			end = begin + len(initial_word)		
			result = []
			result.append([expert])
			result.append(pair[0] + "_NIL")
			result.append(str(begin))
			result.append(str(end))
			result.append(0.5)
	for result in new_resultset:
		if not(len(result) == 1):
			if h == "0":
				pure = result[1].decode("utf-8")
				ne1 = unicodedata.normalize('NFKD', pure).encode('ascii','ignore')
				ne1 = ne1.decode("utf-8")
				bef = ne1.replace("_NIL", "")
				ww = getWholeWord(tweet, bef)
				if len(ww) < 1:
					beg = tweet.find(bef)
					ed = beg + len(bef)
					wwfake = []
					wwfake.append(bef)
					wwfake.append(beg)
					wwfake.append(ed)	
					ww.append(wwfake)
				distance = 1000
				winner = ''
				for w in ww:
					dist = abs(int(result[2])-w[1])
					if(dist < distance):
						distance = dist
						winner = w[0]
				candid0, candid = findID(sol, result[2], result[3])
				something = 1					
				splitted = tweet.split()
				wordbegin = -1
				wordend = -1
				charcounter = 0
				wordcounter = 0
			        foundflag = 0
				candis.add((sid, candid0))
				candis_1.add(candid)
				beginCandidates[candid0] = result[2]
				endCandidates[candid0] = result[3]
				candidates[candid0] = winner
				textid[candid0] = sol
				stateText[candid0] = tweet
				stateText[candid] = tweet
				candidates[candid] = ne1
				candidateHorizon[candid0] = "0.1"
				candidateHorizon[candid] = "1"
				horizon[sol] = '0'
				textid[candid] = sol
				beginCandidates[candid] = result[2]
				endCandidates[candid] = result[3]	
				if not(candid in tempcandidates):
						tempcandidates.append(candid)
						tempcandidatesWONIL.append(candid)					
				hypo = []
				hypo.append(sol)
				hypo.append(candid0)   
				hypo.append('0000' + result[0][0] + candid + sol)         
				hypo.append(candid)
				hypo.append(result[0][0])  
				hypos.append(hypo)
				currentArr = []
				try:
					currentArr = candidatesBy[candid]
				except KeyError:
					pass
				currentArr.append(expert)	
				candidatesBy[candid] = currentArr
				reward = []
				reward.append(sol)
				reward.append(candid0) 
				reward.append(result[0][0])
				reward.append('0000' + result[0][0] + candid + sol)         
				reward.append(candid)
				reward.append(result[4])
				counter = 0
				positive = 0		
				for pair in pairs:
					if (len(get_overlap(ne1,pair[0])) >= (len(ne1) - 1)
					 and len(get_overlap(ne1,pair[0])) >= 1 
					 and ( abs(len(ne1) - len(pair[0])) < 2)):
						if(result[2] == pair[1]) or (pair[1] == -1) or (1 == 1):
							positive = 1
							retrieved += 1
							cb = []
							cb.append(sol)
							cb.append(candid)
							cb.append(1)
							candidateBeliefsArr.append(cb)
							reward.append(1)
							rewards.append(reward)
							break
				if(positive == 0):
					reward.append(0.0)
					retrieved -= 1
					rewards.append(reward)
					cb = []
					cb.append(sol)
					cb.append(candid)
					cb.append(0.33)
					candidateBeliefsArr.append(cb)	
				reward.append(tweetid[sol])
				reward.append(len(pairs))
				try:		
					content = allrewards[sid]
					content.append(reward)
					allrewards[sid] = content
				except:
					content = [reward]
					allrewards[sid] = content
			else:
				candid2 = -1
				candid1 = -1
				res = ""
				res_enc = ""
				allCandidates = state_dic[state]
				ambig = -1
				rx = re.compile('\W+')
				for ca in allCandidates:
					c1 = candidates[ca].lower()
					c2 = result[1].lower()
					c1 = rx.sub(' ', c1)
					c2 = rx.sub(' ', c2)
					if c1 == c2:
						candid1 = ca
						ambig += 1
				if ambig == -1:
					for ca in allCandidates:
						c1 = candidates[ca].lower()
						c2 = result[1].lower()
						c1 = rx.sub(' ', c1)
						c2 = rx.sub(' ', c2)
						if len(get_overlap(c1,c2)) == (len(c2)-1):
							candid1 = ca
							ambig += 1
						if "/" in c1:
							c1 = c1.replace("/", "")
						if "/" in c2:
							c2 = c2.replace("/", "")
						if len(get_overlap(c1,c2)) >= (len(c2)-1):
							candid1 = ca
							ambig += 1
				if ambig == -1:
					print("ambiguous??? " +str(ambig))
					print(result[1])
					for ca in allCandidates:
						print("->" +candidates[ca])
				if ambig == -1:
					continue
				if('_NIL' in result[2]):
					res = result[2]
					res = res.encode("ascii", "ignore")
				else:
					stop = result[2].encode("ascii", "ignore")
					res = getDBpediaURI(stop)
					res = res.decode('unicode_escape')
					res = unicodedata.normalize('NFKD', res).encode('ascii','ignore')
				h = hashlib.sha256(res + str(beginCandidates[candid1]) 
					+ endCandidates[candid1])
				h.hexdigest()
				candis.add((sol, candid1))
				candis_1.add(candid1)
				candid2 = int(h.hexdigest(),base=16)
				candid2 = str(candid2) + tweetid[state]	
				beginCandidates[str(candid2)] = beginCandidates[candid1]							 
				endCandidates[str(candid2)] = endCandidates[candid1]	
				candidates[str(candid2)] = res
				textid[str(candid2)] = sol
				stateText[str(candid2)] = stateText[sol]
				if candid1 == -1 or candid2 == -1:
					continue
				annotatedStateText[candid1] = textWithEntites[state].count("<entity>")	
				if not(str(candid2) in tempcandidates):
					tempcandidates.append(str(candid2))
					tempcandidatesWONIL.append(str(candid2))
				horizon[state] = '1'
				hypo = []
				candidateHorizon[candid1] = "1"
				candidateHorizon[str(candid2)] = "2"
				currentArr = []
				try:
					currentArr = candidatesBy[str(candid2)]
				except KeyError:
					pass
				currentArr.append(result[0][0])	
				candidatesBy[str(candid2)] = currentArr	
				reward = []
				reward.append(state)
				reward.append(candid1) 
				reward.append(result[0][0])
				reward.append(result[0][0])        
				reward.append(str(candid2))
				reward.append(result[5])
				counter = 0
				positive = 0
				if not('_NIL' in res):	
					for pair in pairs:
						if res == pair[3]:
							if positive == 0:
								positive = 1			
						else:
							volde = ""
							latinflag = 0
							try:
								volde = str(res)
							except UnicodeEncodeError:
								latinflag = 1
							if latinflag == 1:
								volde = result[1].encode('windows-1254')
							if getPure(volde) == getPure(pair[3]):
								if positive == 0:
									positive = 1
				else:
					positive = 1
					for pair in pairs:
						reres = res.replace('_NIL', '')
						if reres == pair[0]:
							positive = 0
				reward.append(positive)
				rewards.append(reward)
				reward.append(tweetid[sol])
				reward.append(len(pairs))
				try:
					content = allrewards[state]
					content.append(reward)
					allrewards[state] = content
				except:
					content = [reward]
					allrewards[state] = content
	relevantCandidates = dict()			
	for hypo in rewards:
		if hypo[0] == sol and hypo[0] != hypo[1]:
			relevantCandidates[hypo[1]] = hypo[6]
	for rel in relevantCandidates:
		for algoID in executed[sid]:
			nil = 1
			for hypo in rewards:
				if hypo[1] == rel:
					if hypo[2] == algoID:
						nil = 0
			if nil == 1:
				candid9 = rel + "0000"	
				for candilies in candidates:
					if (candidates[candilies] == candidates[rel] + "_NIL" 
					and textid[candilies] == sol):
						candid9 = candilies
				if not(candid9 in tempcandidates):
					tempcandidates.append(candid9)
				beginCandidates[candid9] = beginCandidates[rel]
				endCandidates[candid9] = endCandidates[rel]
				candidates[candid9] = candidates[rel] + "_NIL"
				textid[candid9] = textid[rel]
				stateText[candid9] = tweet
				candidateHorizon[candid9] = candidateHorizon[rel]
				currentArr = []
				try:
					currentArr = candidatesBy[candid9]
				except KeyError:
					pass
				currentArr.append(algoID)	
				candidatesBy[candid9] = currentArr
				candis_1.add(candid9)
				reward = []
				reward.append(sol)
				reward.append(rel) 
				reward.append(algoID)
				reward.append('0000' + algoID + rel + sol)         
				reward.append(candid9)
				reward.append(0.5)
				receivedReward = 1.0
				for pair in pairs:
					if (len(get_overlap(candidates[rel],pair[0])) >= (len(pair[0]) - 1)
					 and len(get_overlap(candidates[rel],pair[0]))) >= 1:
						receivedReward = 0.0
				reward.append(receivedReward)
				if True:
					print("NOT NOT NOT: (" +str(algoID) + ") " +str(candidates[candid9])
					 + " - " +str(receivedReward))
					rewards.append(reward)
					content = allrewards[sid]
					content.append(reward)
					allrewards[sid] = content
					candidateBeliefs[str(candid9)] = receivedReward
					hasReward['0000' + algoID + rel + sol] = receivedReward
					reward.append(tweetid[sol])
					reward.append(len(pairs))
	if h == "0":
		stateCandis[sol + "2222222222"] = candis_1
		reward2 = []
		reward2.append(sol)
		reward2.append(sol)	
		reward2.append(expert)
		reward2.append(expert)
		reward2.append(sol + "2222222222")
		reward2.append(0.5)
		finalrew = 0.0
		if float(retrieved) >= 0.0 and len(pairs) > 0:
			finalrew = float(1) - float(( abs(float(len(pairs)) - float(retrieved))
			 / float(len(pairs)) ))
		reward2.append(finalrew)
		reward2.append(sol)
		reward2.append(len(pairs))
		rewards.append(reward2)
	return candis

def explore_online_hedge():
	global budget
	global bounds
	global base
	global dataset_mode
	global datapoints
	global dependencies
	mode = "train"
	datapoints = formatDataset(dataset_mode)
	print("LEN " +str(len(datapoints)))
	datapoints_set = random.sample(datapoints, limit) # in case of Spotlight: 59
	past_points = dict()
	H = ['0', '1']
	t = 1
	#####init dependencies #####################################################
	dependencies.append("LiveDOM")
	for expert in experts_dic["0"]:
		expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
		for b in behav["DOI"]:
			dependencies.append(("LiveDOI", b, expert))
	for expert in experts_dic["1"]:
		expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
		for b in behav["DOI"]:
			dependencies.append(("LiveDOI", b, expert))
	for expert in experts_dic["1"]:
		for b in behav["DOR"]:
			dependencies.append(("LiveDOR", b, expert))
	initWeights(experts_dic['0'], metasims_dic['0.1'], dependencies, str(t))
	initWeights(experts_dic['1'], metasims_dic['1'], dependencies, str(t))
	#####start learning #######################################################
	for point in datapoints_set:
		tweetid[point] = point
		text = datapoints[point][0]
		textid[point] = point
		labels = datapoints[point][3]
		candidateHorizon[point] = "0"
		candidates[point] = text
		stateText[point] = text
		weighted_samples = dict()
		weighted_samples['0'] = [[[point, point]]]
		for h in H:
			print("+-+-+-+-+- h=" +str(h) +" +-+-+-+-+-+-")
			for ws in weighted_samples[h]:
				for candi in ws:
					doMS(candi[1], h, metasims_dic[h])
			getNeigbs()
			results1 = dict()
			if h == "0":
				results1 = apriori_assess(experts_dic[h], experts_dic[str(int(h)+1)], 
					weighted_samples[h], metasims_dic[h], h, bounds)
			else:
				results1 = apriori_assess(experts_dic[h], [], weighted_samples[h], 
					metasims_dic[h], h, bounds)
			candis, chosen_experts = choose_and_execute(budget[h], results1, h, str(t), 
				metasims_dic[h])
			if h == "0":
				h = "0.1"
				for ca in candis:
					doMS(ca[1],h, metasims_dic[h])
			getNeigbs()
			results2 = dict()
			us = None
			pds = None
			results_dep = None
			Distr = None
			if h == "0" or h == "0.1":
				temph = "0"
				results2, us, pds, results_dep, Distr = dealWithWeights_pre(candis, 
					experts_dic[h], experts_dic[str(int(temph)+1)], metasims_dic[h],
					 str(t), bounds, h, 0.2, "online", dependencies, chosen_experts)
			else:
				results2, us, pds, results_dep, Distr = dealWithWeights_pre(candis,
				 experts_dic[h], [], metasims_dic[h], str(t), bounds, h, 0.2, "online",
				  dependencies, chosen_experts)
			newh = -1
			if float(h) == 0.1:
				newh = 1
			else:
				newh = int(h) + 1
			weighted_samples[str(newh)], winna = getNewSamples(metasims_dic[h], h, 1,
			 0.2, point, str(t), bounds, results2, mode)
			dealWithWeights_post(us, pds, t, metasims_dic[h], mode, dependencies, 
				experts_dic[h], winna, chosen_experts)
		past_points[point] = 1	
		if len(past_points) == limit:
			break
		if len(past_points) == lower_bound:
			mode = "test"	
		t += 1

def setupBounds():
	global bounds
	global lower_bound
	bounds[('SameLingualType', "0")] = float(lower_bound) / 16.0 
	bounds[('CandidateWordEmbeddings', "0")] = float(lower_bound) / 35.0
	bounds[('CandidateTextMetaSimilarity', "0")] = float(lower_bound) / 35.0
	bounds[('StateTextLengthMetaSimilarity', "0")] = float(lower_bound) / 5.0 
	bounds[('StateCandidateWordEmbeddings', "0")] = float(lower_bound)  / 40.0
	bounds[('StateSameLingualType', "0")] = float(lower_bound) / 10.0
	bounds[('StateTextExtraChars', "0")] = float(lower_bound)

	bounds[('SameLingualType', "0.1")] = bounds[('SameLingualType', "0")]
	bounds[('CandidateWordEmbeddings', "0.1")] = bounds[('CandidateWordEmbeddings', "0")]
	bounds[('CandidateTextMetaSimilarity', "0.1")] = bounds[('CandidateTextMetaSimilarity', "0")]
	bounds[('StateTextLengthMetaSimilarity', "0.1")] = bounds[('StateTextLengthMetaSimilarity', "0")]
	bounds[('StateCandidateWordEmbeddings', "0.1")] = bounds[('StateCandidateWordEmbeddings', "0")]
	bounds[('StateSameLingualType', "0.1")] = bounds[('StateSameLingualType', "0")]
	bounds[('StateTextExtraChars', "0.1")] = bounds[('StateTextExtraChars', "0")]

	bounds[('SameLingualType', "1")] = bounds[('SameLingualType', "0")]
	bounds[('CandidateWordEmbeddings', "1")] = bounds[('CandidateWordEmbeddings', "0")]
	bounds[('CandidateTextMetaSimilarity', "1")] = bounds[('CandidateTextMetaSimilarity', "0")]
	bounds[('StateTextLengthMetaSimilarity', "1")] = bounds[('StateTextLengthMetaSimilarity', "0")]
	bounds[('StateCandidateWordEmbeddings', "1")] = bounds[('StateCandidateWordEmbeddings', "0")]
	bounds[('StateSameLingualType', "1")] = bounds[('StateSameLingualType', "0")]
	bounds[('StateTextExtraChars', "1")] = bounds[('StateTextExtraChars', "0")]

def main(argv=None):
	if argv is None:
		argv = sys.argv
	global wv
	global lower_bound
	global limit
	global budget
	global dataset_mode
	lower_bound = int(argv[2])
	limit = int(argv[3])
	setupBounds()
	budget["0"] = int(argv[4])
	budget["0.1"] = int(argv[5])
	budget["1"] = int(argv[6])
	dataset_mode = argv[7]
	logging.getLogger("requests").setLevel(logging.WARNING)
	logging.getLogger("urllib3").setLevel(logging.WARNING)
	options = process_args(argv[1:2])
	try:
		wv = wvlib.load(options.vectors, max_rank=options.max_rank)
     	  	wv = wv.normalize()
	except Exception, e:
		print >> sys.stderr, 'Error: %s' % str(e)
	explore_online_hedge()

if __name__=='__main__':
    main(sys.argv)