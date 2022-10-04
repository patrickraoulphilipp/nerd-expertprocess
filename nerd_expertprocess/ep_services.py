import requests
import json
from rdflib.graph import Graph

from nerd_expertprocess.ep_utils import get_feature_id

def execute_AGDISTIS(algoid, aTweet):
    parameterizedFeatures = []
    parameterizedFeatures.append(["CoherenceEntityLinks",
     algoid+get_feature_id("CoherenceEntityLinks"), "1"])
    parameterizedFeatures.append(["MentionPrior",
     algoid+get_feature_id("MentionPrior"), "1"])
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

def execute_AIDA_tagger(algoid, aTweet):
    parameterizedFeatures = []
    parameterizedFeatures.append(["CoherenceEntityLinks",
     algoid+get_feature_id("CoherenceEntityLinks"), "1"])
    parameterizedFeatures.append(["MentionPrior", 
        algoid+get_feature_id("MentionPrior"), "1"])
    parameterizedFeatures.append(["SyntacticBasedContextualEntityFrequencyOccurence",
     algoid+get_feature_id("SyntacticBasedContextualEntityFrequencyOccurence"), "1"])
    parameterizedFeatures.append(["MentionInfoSourceWithRepresentativeSentences",
     algoid+get_feature_id("MentionInfoSourceWithRepresentativeSentences"), "1"])
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

def execute_FOX(algoid, tweet, ensembleids):
    parameterizedFeatures = []
    parameterizedFeatures.append(["StanfordNEROutput", algoid+get_feature_id("StanfordNEROutput"), "1"])
    parameterizedFeatures.append(["IllinoisNETOutput", algoid+get_feature_id("IllinoisNETOutput"), "1"])
    parameterizedFeatures.append(["BalieOutput", algoid+get_feature_id("BalieOutput"), "1"])
    parameterizedFeatures.append(["OpenNLPOutput", algoid+get_feature_id("OpenNLPOutput"), "1"])
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
    r = requests.post('http://fox-demo.aksw.org/call/ner/entities', data=json.dumps(data), headers=headers)
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

def execute_stanford_NER(algoid, tweet):
    parameterizedFeatures = []
    parameterizedFeatures.append(["SyntacticTextFeature", algoid+get_feature_id("SyntacticTextFeature"), "1"])
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
    result = r.text
    g = Graph()
    print(result)
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

def execute_spotlight_tagger(algoid, tweet, combis, candidates, begin_candidates, 
    confidence, support):
    tweet = tweet.replace("&", "&amp;")
    tweet = tweet.replace("\"", "&quot;" )
    parameterizedFeatures = []
    parameterizedFeatures.append(["WordPhrases", 
        algoid+get_feature_id("WordPhrases"), "1"])
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
            .format(enc, int(begin_candidates[cid])))
    tweet = tweet + '</annotation>'
    results = []
    data = {'text': '%s' % tweet, 'confidence': '%i' % confidence, 
    'support': '%i' % support}
    headers = {'accept': 'application/json'}
    r = requests.post('http://aifb-ls3-remus.aifb.kit.edu:2225/rest/disambiguate', 
        data=data, headers=headers)
    res = r.json()
    try:
        res['Resources']
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

def execute_spotlight_spotter(algoid, tweet):
    parameterizedFeatures = []
    parameterizedFeatures.append(["WordPhrases", 
        algoid+get_feature_id("WordPhrases"), "1"])
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
        res['Resources']
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