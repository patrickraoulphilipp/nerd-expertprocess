import re
import hashlib
import unicodedata

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_services import *
from nerd_expertprocess.ep_utils import *

def execute(chosen, h):
    candis = set()
    candis_1 = set()
    expert = chosen[0]
    sid = chosen[1]
    tweet = ep_state.state_text[sid]
    pairs = ep_state.datapoints[ep_state.tweetid[sid]][3]
    resultset = []
    try:
        if expert == "01":
            resultset = get_from_stash("01",tweet)
            if resultset == None:
                resultset = execute_FOX("01", tweet, ["04"])
        if expert == "02":
            resultset = get_from_stash("02",tweet)
            if resultset == None:
                resultset = execute_spotlight_spotter("02", tweet)
        if expert == "03":
            resultset = get_from_stash("03",tweet)
            if resultset == None:
                resultset = execute_stanford_NER("03", tweet)
        if expert == "11":
            if "<entity>" in ep_state.text_with_entites[sid]:
                resultset = get_from_stash("11", ep_state.text_with_entites[sid])		
                if resultset == None:
                    resultset = execute_AGDISTIS('11', ep_state.text_with_entites[sid])
        if expert == "12":
            if "<entity>" in ep_state.text_with_entites[sid]:
                resultset = get_from_stash("12",ep_state.text_with_entites[sid])
                if resultset == None:
                    resultset = execute_AIDA_tagger('12', ep_state.text_with_entites[sid])
        if expert == "13":
            if "<entity>" in ep_state.text_with_entites[sid]:
                resultset = get_from_stash("13", ep_state.text_with_entites[sid])
                if resultset == None:
                    resultset = execute_spotlight_tagger('13', ep_state.text_with_entites[sid],
                    ep_state.state_dic[sid], ep_state.candidates, ep_state.begin_candidates, 0.3, 0)
    except Exception as e:
        return None
    try:
        prior = ep_state.executed[sid]
        prior.append(expert)
        ep_state.executed[sid] = prior
    except:
        prior = [expert]
        ep_state.executed[sid] = prior
    tempcandidates = []
    tempcandidatesWONIL = []
    sol = sid
    state = sid
    retrieved = 0
    new_resultset = []
    for result in resultset:
        if not(len(result) == 1):
            new_resultset.append(result)
    for pair in pairs:
        found = False
        for result in resultset:
            if not(len(result) == 1):
                ne1 = result[1]
                if (len(get_overlap(ne1,pair[0])) >= (len(pair[0]) - 1)
                 and len(get_overlap(ne1,pair[0]))) >= 1:
                    found = True
                    break
        if not(found) and int(expert) < 9:
            ww = find_and_clean(tweet, pair[0])
            if len(ww) > 0:
                initial_word = None
                begin = -1
                end = -1
                initial_word = ww[0][0]
                begin = ww[0][1]
                end = begin + len(initial_word)		
                result = [
                    [expert],
                    pair[0] + "_NIL",
                    str(begin),
                    str(end),
                    default_confidence
                ]
    for result in new_resultset:
        if not(len(result) == 1):
            if h == "0":
                pure = result[1]
                ne1 = unicodedata.normalize('NFKD', pure).encode('ascii','ignore')
                ne1 = ne1.decode("utf-8")
                bef = ne1.replace("_NIL", "")
                ww = find_and_clean(tweet, bef)
                if len(ww) < 1:
                    beg = tweet.find(bef)
                    ed = beg + len(bef)
                    wwfake = [
                        bef,
                        beg,
                        ed
                    ]
                    ww.append(wwfake)
                distance = 1000
                winner = ''
                for w in ww:
                    dist = abs(int(result[2])-w[1])
                    if(dist < distance):
                        distance = dist
                        winner = w[0]
                candid0, candid = find_id(sol, result[2], result[3])
                candis.add((sid, candid0))
                candis_1.add(candid)
                ep_state.begin_candidates[candid0] = result[2]
                ep_state.end_candidates[candid0] = result[3]
                ep_state.candidates[candid0] = winner
                ep_state.text_id[candid0] = sol
                ep_state.state_text[candid0] = tweet
                ep_state.state_text[candid] = tweet
                ep_state.candidates[candid] = ne1
                ep_state.candidate_horizon[candid0] = "0.1"
                ep_state.candidate_horizon[candid] = "1"
                ep_state.horizon[sol] = '0'
                ep_state.text_id[candid] = sol
                ep_state.begin_candidates[candid] = result[2]
                ep_state.end_candidates[candid] = result[3]	
                if not(candid in tempcandidates):
                        tempcandidates.append(candid)
                        tempcandidatesWONIL.append(candid)					
                hypo = [
                    sol,
                    candid0,
                    '0000' + result[0][0] + candid + sol,
                    candid,
                    result[0][0]
                ]
                ep_state.hypos.append(hypo)
                currentArr = []
                try:
                    currentArr = ep_state.candidates_by[candid]
                except KeyError:
                    pass
                currentArr.append(expert)	
                ep_state.candidates_by[candid] = currentArr
                reward_value = 0
                for pair in pairs:
                    if (len(get_overlap(ne1,pair[0])) >= (len(ne1) - 1)
                     and len(get_overlap(ne1,pair[0])) >= 1 
                     and ( abs(len(ne1) - len(pair[0])) < 2)):
                        if(result[2] == pair[1]) or (pair[1] == -1) or (1 == 1):
                            retrieved += 1
                            reward_value = 1
                            break
                if(reward_value == 0):
                    retrieved -= 1
                reward = [
                    sol,
                    candid0,
                    result[0][0],
                    '0000' + result[0][0] + candid + sol,
                    candid,
                    result[4],
                    reward_value,
                    ep_state.tweetid[sol],
                    len(pairs)
                ]
                ep_state.rewards.append(reward)
                try:		
                    content = ep_state.all_rewards[sid]
                    content.append(reward)
                    ep_state.all_rewards[sid] = content
                except:
                    content = [reward]
                    ep_state.all_rewards[sid] = content
            else:
                candid2 = -1
                candid1 = -1
                res = ""
                allCandidates = ep_state.state_dic[state]
                ambig = -1
                rx = re.compile('\W+')
                for ca in allCandidates:
                    c1 = ep_state.candidates[ca].lower()
                    c2 = result[1].lower()
                    c1 = rx.sub(' ', c1)
                    c2 = rx.sub(' ', c2)
                    if c1 == c2:
                        candid1 = ca
                        ambig += 1
                if ambig == -1:
                    for ca in allCandidates:
                        c1 = ep_state.candidates[ca].lower()
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
                    continue
                if('_NIL' in result[2]):
                    res = result[2]
                    res = res.encode("ascii", "ignore")
                else:
                    stop = result[2].encode("ascii", "ignore")
                    res = get_dbpedia_uri(stop)
                
                candid_id = res + str(ep_state.begin_candidates[candid1]) + str(ep_state.end_candidates[candid1])
                candid_id = unicodedata.normalize('NFKD', candid_id).encode('ascii','ignore')
                h = hashlib.sha256(candid_id)
                h.hexdigest()
                candis.add((sol, candid1))
                candis_1.add(candid1)
                candid2 = int(h.hexdigest(),base=16)
                candid2 = str(candid2) + ep_state.tweetid[state]	
                ep_state.begin_candidates[str(candid2)] = ep_state.begin_candidates[candid1]							 
                ep_state.end_candidates[str(candid2)] = ep_state.end_candidates[candid1]	
                ep_state.candidates[str(candid2)] = res
                ep_state.text_id[str(candid2)] = sol
                ep_state.state_text[str(candid2)] = ep_state.state_text[sol]
                if candid1 == -1 or candid2 == -1:
                    continue
                if not(str(candid2) in tempcandidates):
                    tempcandidates.append(str(candid2))
                    tempcandidatesWONIL.append(str(candid2))
                ep_state.horizon[state] = '1'
                ep_state.candidate_horizon[candid1] = "1"
                ep_state.candidate_horizon[str(candid2)] = "2"
                currentArr = []
                try:
                    currentArr = ep_state.candidates_by[str(candid2)]
                except KeyError:
                    pass
                currentArr.append(result[0][0])	
                ep_state.candidates_by[str(candid2)] = currentArr	
                reward_value = 0
                if not('_NIL' in res):	
                    for pair in pairs:
                        if res == pair[3]:
                            if reward_value == 0:
                                reward_value = 1			
                        else:
                            volde = ""
                            latinflag = 0
                            try:
                                volde = str(res)
                            except UnicodeEncodeError:
                                latinflag = 1
                            if latinflag == 1:
                                volde = result[1].encode('windows-1254')
                            if get_pure(volde) == get_pure(pair[3]):
                                if reward_value == 0:
                                    reward_value = 1
                else:
                    reward_value = 1
                    for pair in pairs:
                        reres = res.replace('_NIL', '')
                        if reres == pair[0]:
                            reward_value = 0
                reward = [
                    state,
                    candid1,
                    result[0][0],
                    result[0][0],
                    str(candid2),
                    result[5],
                    reward_value,
                    ep_state.tweetid[sol],
                    len(pairs)
                ]
                ep_state.rewards.append(reward)
                try:
                    content = ep_state.all_rewards[state]
                    content.append(reward)
                    ep_state.all_rewards[state] = content
                except:
                    content = [reward]
                    ep_state.all_rewards[state] = content
    relevantCandidates = dict()			
    for hypo in ep_state.rewards:
        if hypo[0] == sol and hypo[0] != hypo[1]:
            relevantCandidates[hypo[1]] = hypo[6]
    for rel in relevantCandidates:
        for algoID in ep_state.executed[sid]:
            nil = 1
            for hypo in ep_state.rewards:
                if hypo[1] == rel:
                    if hypo[2] == algoID:
                        nil = 0
            if nil == 1:
                candid9 = rel + "0000"	
                for candilies in ep_state.candidates:
                    if (ep_state.candidates[candilies] == ep_state.candidates[rel] + "_NIL" 
                    and ep_state.text_id[candilies] == sol):
                        candid9 = candilies
                if not(candid9 in tempcandidates):
                    tempcandidates.append(candid9)
                ep_state.begin_candidates[candid9] = ep_state.begin_candidates[rel]
                ep_state.end_candidates[candid9] = ep_state.end_candidates[rel]
                ep_state.candidates[candid9] = ep_state.candidates[rel] + "_NIL"
                ep_state.text_id[candid9] = ep_state.text_id[rel]
                ep_state.state_text[candid9] = tweet
                ep_state.candidate_horizon[candid9] = ep_state.candidate_horizon[rel]
                currentArr = []
                try:
                    currentArr = ep_state.candidates_by[candid9]
                except KeyError:
                    pass
                currentArr.append(algoID)	
                ep_state.candidates_by[candid9] = currentArr
                candis_1.add(candid9)
                reward_value = 1.0
                for pair in pairs:
                    if (len(get_overlap(ep_state.candidates[rel],pair[0])) >= (len(pair[0]) - 1)
                     and len(get_overlap(ep_state.candidates[rel],pair[0]))) >= 1:
                        reward_value = 0.0
                reward = [
                    sol,
                    rel,
                    algoID,
                    '0000' + algoID + rel + sol,
                    candid9,
                    default_confidence,
                    reward_value,
                    ep_state.tweetid[sol],
                    len(pairs)
                ]
                ep_state.rewards.append(reward)
                content = ep_state.all_rewards[sid]
                content.append(reward)
                ep_state.all_rewards[sid] = content
                ep_state.candidate_beliefs[str(candid9)] = reward_value
    if h == "0":
        reward_value = 0.0
        if float(retrieved) >= 0.0 and len(pairs) > 0:
            reward_value = float(1) - float(( abs(float(len(pairs)) - float(retrieved))
             / float(len(pairs)) ))
        reward = [
            sol,
            sol,
            expert,
            expert,
            sol + "2222222222",
            default_confidence,
            reward_value,
            sol,
            len(pairs)
        ]
        ep_state.rewards.append(reward)
    return candis