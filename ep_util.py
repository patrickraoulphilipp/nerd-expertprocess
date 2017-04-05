import io
import numpy as np
import unicodedata
import nltk
import requests
import json
import random
import cPickle
import re
import difflib
from nltk.corpus import stopwords
from nltk.util import ngrams
from textblob import TextBlob
from rdflib.graph import Graph
from difflib import SequenceMatcher

def countExtras(tweet):
	counthashtext = tweet.count('#')
	countattext = tweet.count("@")	
	return counthashtext + countattext + 1	

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