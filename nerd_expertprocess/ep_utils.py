from nltk.util import ngrams
import difflib
import pickle
import re

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_config import *

def get_precision(window=None):
    window = -1 * window if window is not None else None
    res = 1.
    if sum(ep_state.Y_Pred[window:]) > 0:
        res = sum(ep_state.Y_Label[window:]) / sum(ep_state.Y_Pred[window:])
    return res

def get_recall(window=None):
    window = -1 * window if window is not None else None
    res = 1.
    if sum(ep_state.Y_Labels[window:]) > 0:
        res = sum(ep_state.Y_Label[window:]) / sum(ep_state.Y_Labels[window:]) 
    return res

def get_pure(entity):
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

def get_dbpedia_uri(entity):
    entity = entity.decode("utf-8") 
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

def get_feature_id(feature):
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

def find_id(sol, begin, end):
    uptop = 3
    rangePlus = range(uptop)
    rangeMinus = []
    for ra in rangePlus:
        b = -1 * ra
        rangeMinus.append(b)
    completeRange = rangeMinus + list(rangePlus)
    reducedRange = set(completeRange)	
    for ra in reducedRange:
        newBegin = str(int(begin)+ra)
        newEnd = str(int(end)+ra)	
        candid = sol + newBegin + newEnd	
        candid0 = "0000" + sol + newBegin + newEnd
        try:
            ep_state.candidates[candid0,candid]
            return candid
        except:
            pass
    for ra in reducedRange:
        newBegin = begin
        newEnd = str(int(end)+ra)	
        candid = sol + newBegin + newEnd	
        candid0 = "0000" + sol + newBegin + newEnd
        try:
            ep_state.candidates[candid0,candid]
            return candid
        except:
            pass
    for ra in reducedRange:
        newBegin = str(int(begin)+ra)
        newEnd = end	
        candid = sol + newBegin + newEnd	
        candid0 = "0000" + sol + newBegin + newEnd
        try:
            ep_state.candidates[candid0,candid]
            return candid
        except:
            pass
    candid = sol + begin + end	
    candid0 = "0000" + sol + begin + end
    return candid0, candid	

def find_and_clean(tweet, ne):
    ww = None
    possibilities = ["", "#", ",", "."]
    for po in possibilities:
        ww = find_word(tweet, ne, po)
        if len(ww) > 0:
            break
    if len(ww) == 0:
        print("nothing for " +str(ne) + "IN " +str(tweet))
    return ww

def find_word(tweet, ne, to_remove):
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

def get_overlap(s1, s2):
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, _, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]

def get_from_stash(algo, text):
    res = None
    text = text.replace("/","")
    if len(text) > 200:
        text = text[0:200]
    path = STASH_PATH +str(algo) + '_' + text + '.txt'
    res = pickle.load( open( path, "rb" ) )
    return res

def annotateCandidatesInText(
        candidates, 
        begin_candidates,
        end_candidates, 
        subset, 
        text
):	
    counter = 0
    starts = []
    newText = text	
    for s in subset:
        cands = candidates[s]
        if not('_NIL' in cands):
            counter +=1
            if (text.count(cands) == 1) and (cands != "NIL" ):
                newText = newText.replace(cands,'<entity>' + cands + '</entity>')
            else:
                allindexes = [m.start() for m in re.finditer(candidates[s], text)]
                for ind in allindexes:
                    if(str(ind) == begin_candidates[s]):
                        if counter == 1:
                            newText = newText[:int(begin_candidates[s])] + newText[int(begin_candidates[s]):int(end_candidates[s])].replace(cands,'<entity>' + cands + '</entity>') + newText[int(end_candidates[s]):]
                        else:
                            toAdd = 0
                            for start in starts:
                                if int(begin_candidates[s]) > int(start):
                                    toAdd += 17
                            newText = newText[:int(begin_candidates[s])+toAdd] + newText[int(begin_candidates[s])+toAdd:int(end_candidates[s])+toAdd].replace(cands,'<entity>' + cands + '</entity>') + newText[int(end_candidates[s])+toAdd:]
            starts.append(begin_candidates[s])				
    return newText

def getCRs():
    for rew in ep_state.rewards:
        ep_state.candidate_rewards[rew[4]] = rew[6]

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