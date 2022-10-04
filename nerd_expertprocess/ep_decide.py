import math
import random

from nerd_expertprocess.ep_learn import *
from nerd_expertprocess.ep_execute import execute

def sample_candidates(
        h, 
        currentTweetID, 
        t, 
        results, 
        fail
):
    getCRs()
    some_collector = dict()
    reward_collector = dict()
    by_exp = dict()
    for result in results:
        rew = result[0]
        regressors = result[1]
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
    all_best = []
    for best in reward_collector:
        all_best.extend(reward_collector[best])
    for best in reward_collector:
        rew = calculate_adapted_reward(
                  reward_collector[best], 
                  t, 
              )
        turned_collector[best] = rew
    for best in turned_collector:
        rew = turned_collector[best]
    normalizer = 0.0
    for best1 in turned_collector:
        final_collector[best1] = turned_collector[best1]
        normalizer += float(final_collector[best1])
    clusters = []
    for best in final_collector:
        p = best
        r2 = range(int(ep_state.begin_candidates[p]), int(ep_state.end_candidates[p]))
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
    for _ in iterations:	
        choice = []	
        for cluster in clusters:
            begin = 0
            end = 0
            tochoose = []
            todraw =  []
            for part in cluster:
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
                        if begin > int(ep_state.begin_candidates[part[1]]):
                            begin = int(ep_state.begin_candidates[part[1]])
                        if end < int(ep_state.end_candidates[part[1]]):
                            end = int(ep_state.end_candidates[part[1]])						
                        possies[part[1]] = part[1]
                        todraw.append([part[1], part_rew])
            todraw.extend(tochoose)
            iters = range(40)
            foundthebitch = 1
            for _ in iters:
                proposition = weighted_choice_hedge(todraw)	
                for prior in choice:
                    range1 = []
                    range2 = []
                    try:
                        range1 = range(int(ep_state.begin_candidates[prior[0]]), 
                            int(ep_state.end_candidates[prior[0]])+1)
                    except KeyError:
                        continue
                    try:
                        range2 = range(int(ep_state.begin_candidates[proposition[0]]), 
                            int(ep_state.end_candidates[proposition[0]])+1)
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
        weighted_choices.append(choice)
        for ch in choice:
            if not '_NIL' in ch[0]:
                skinnychoice.append(ch[0])
        choices.append(skinnychoice)
    combiIDcounter = 0
    regrets = []
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
    mychoice = random.sample(choices,1)
    mychoice = mychoice[0]
    if not fail:
        for possi in possies:
            flag = -1
            for ch in mychoice:
                if possi == ch:
                    flag = 1
            if flag == -1:
                ep_state.Y_Pred_global.append(0)
            else:
                ep_state.Y_Pred_global.append(1)
            res = round(ep_state.candidate_rewards[possi])
            ep_state.Y_Label_global.append(res)
        if h == "1":
            ep_state.Y_Label_count += len(ep_state.datapoints[currentTweetID][3])
            ep_state.Y_Labels.append(len(ep_state.datapoints[ep_state.tweetid[currentTweetID]][3]))
            for part in mychoice:
                if not("_NIL" in ep_state.candidates[part]):
                    ep_state.Y_Pred.append(1.0)
                    ep_state.Y_Label.append(ep_state.candidate_rewards[part])
                    ep_state.Y_Label_text.append(ep_state.candidates[part])
                    ep_state.Y_Pred_count += int(ep_state.candidate_rewards[part])
                    ep_state.Y_Preds.append(ep_state.candidates[part])
        else:
            ep_state.Y_Label_count_1 += len(ep_state.datapoints[currentTweetID][3])
            for part in mychoice:
                try:
                    ep_state.chosen_exps.append(by_exp[part])
                except:
                    ep_state.chosen_exps.append("Add_NIL")
                if not("_NIL" in ep_state.candidates[part]):
                    ep_state.Y_Pred_1.append(1.0) #TODO just use "1"?!
                    ep_state.Y_Label_1.append(ep_state.candidate_rewards[part])
                    ep_state.Y_Label_text_1.append(ep_state.candidates[part])
                    ep_state.Y_Pred_count_1 += int(ep_state.candidate_rewards[part])
    combiIDcounter = 0	
    state_choices = []
    for combi in [mychoice]:
        if len(combi) == 0:
            continue
        newStateID = currentTweetID + str(combiIDcounter)
        newText = annotateCandidatesInText(ep_state.candidates, ep_state.begin_candidates, ep_state.end_candidates, 
            combi, ep_state.state_text[currentTweetID])		
        ep_state.text_with_entites[newStateID] = newText
        newState = []
        for p in combi:
            ep_state.candidate_horizon[p] = "1"
            if not("_NIL" in ep_state.candidates[p]):
                newState.append([newStateID, p])
        state_choices.append(newState)
        ep_state.state_dic[newStateID] = combi
        ep_state.state_text[newStateID] = ep_state.state_text[currentTweetID]
        ep_state.tweetid[newStateID] = currentTweetID
        combiIDcounter += 1
    return state_choices	

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
    to_stay = []
    for choice in choices:
        c, w = choice
        prob = float(w) / float(total)	
        prob = mapIntervalsBack(prob)
        if float(prob) < global_min_prob_hedge:
            new_weight = global_min_prob_hedge 
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

def choose_and_execute(
        budget, 
        weights, 
        h, 
        t, 
):
    D = dict()
    chosen_experts = []
    for w in weights:
        D[w[2]] = 1.0
    candis = set()
    for _ in range(budget):
        if len(weights) > 0:
            pd, _ = make_pd_hedge(weights, D, t)
            chosen = weighted_choice_hedge(pd)
            real_probs = get_real_probs(pd)
            temp_candis = execute(chosen[0], h)
            if temp_candis is None:
                ep_state.failed_experts += 1
                continue # the expert service was not available
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