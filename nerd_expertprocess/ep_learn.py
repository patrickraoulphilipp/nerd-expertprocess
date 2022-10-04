import math

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_metadeps import *
from nerd_expertprocess.ep_config import *
from nerd_expertprocess.ep_utils import *

def init_weights(experts, metasims, dependencies, t):
    for expert in experts:
        for ms in metasims:
            for dependency in dependencies:
                ep_state.expert_weight_dic[(expert, ms, dependency, t)] = 1.0

def default_weight_update(
        expert, 
        metasims, 
        dependencies, 
        t
):
    for ms in metasims:
        for dependency in dependencies:
            ep_state.expert_weight_dic[(expert, ms, dependency, t)] = ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)-1))] 

def deal_with_weights_preselection(
        weighted_samples, 
        experts, experts2, 
        metasims, 
        t, 
        bounds, 
        h, 
        threshold, 
):
    uniqueStates = set()
    memorizer = set()
    results = dict()
    D = dict()
    give_back = []
    for candidate_state in weighted_samples: 
        D[(candidate_state[0], candidate_state[1])] = 1.0
        for expert in experts:
            result = calculate_relatedness_md(
                          expert, 
                          candidate_state[0], 
                          candidate_state[1], 
                          metasims, 
                          h, 
                          threshold, 
                          bounds
                     )
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
                    result2 = calculate_pairwise_md(expert1, expert2, candidate_state[0], 
                        candidate_state[1], metasims, h, threshold, bounds)
                    try:
                        current = results[(expert1, candidate_state[0], candidate_state[1])]
                        current.append([result2, ("LiveDOI", expert2)])
                        results[(expert1, candidate_state[0], candidate_state[1])] = current
                    except:
                        results[(expert1, candidate_state[0], candidate_state[1])] = [[result2, 
                        ("LiveDOI", expert2)]]
            for expert2 in experts2:
                result3 = calculate_robustess_md(expert1, expert2, candidate_state[0], candidate_state[1],
                 metasims, h, threshold, bounds)
                try:
                    current = results[(expert1, candidate_state[0], candidate_state[1])]
                    current.append([result3, ("LiveDOR", expert2)])
                    results[(expert1, candidate_state[0], candidate_state[1])] = current
                except:
                    results[(expert1, candidate_state[0], candidate_state[1])] = [[result3,
                     ("LiveDOR", expert2)]]
    if h != "0":
        pd, pd2 = make_pd_hedge(results, D, t)
        real_pd = get_real_probs(pd)
        pds = [real_pd,pd2]
        for us in uniqueStates:
            allRews = []
            try:
                allRews = ep_state.all_rewards[us]
            except:
                continue
            for rew in allRews:
                if rew[0] == rew[1]:
                    continue
                memorizer.add(rew[2])
                give_back.append([rew, results[rew[2], rew[0], rew[1]]])
    return give_back, uniqueStates, pds

def update_weights(
        expert, 
        total, 
        state, 
        candi, 
        retrieved_reward, 
        pds, 
        metasims, 
        t, 
        dependencies, 
):
    pd, pd2 = pds
    if float(total) == 0.0:
        total = 0.1
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
                    ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
                except:
                    ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)))]
            else:
                reward = ( 1 - abs( float(retrieved_reward) - float(prediction)  ))			
                reward = mapIntervals(reward)
                reward = float(reward) * float(density)
                p_expert += p_min_global
                reward = float(reward) / float(p_expert) 
                try:
                    ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
                except:
                    ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)))]
                try:	
                    ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = float(ep_state.expert_weight_dic[(expert, ms, dependency, str(t))]) * float(math.exp(reward * p_min_global))
                except:	
                    try:
                        ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = ep_state.expert_weight_dic[(expert, ms, dependency, str(t))]						
                    except:
                        ep_state.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = math.exp(100)

def deal_with_weights_postselection(
        uniqueStates, 
        pds, 
        t, 
        metasims, 
        mode, 
        dependencies, 
        experts, 
):
    memorizer = set()
    pd, _ = pds
    total = sum(w for c, w in pd)
    if mode == "train":
        for us in uniqueStates:
            collector = dict()
            allRews = []
            try:
                allRews = ep_state.all_rewards[us]
            except:
                continue
            for rew in allRews:
                if rew[0] == rew[1]:
                    continue
                update_weights(
                    rew[2], 
                    total, 
                    rew[0], 
                    rew[1], 
                    rew[6], 
                    pds, 
                    metasims, 
                    t, 
                    dependencies, 
                )
                memorizer.add(rew[2])
                try:
                    collector[rew[2]].append(rew)
                except:
                    collector[rew[2]] = []
                    collector[rew[2]].append(rew)
    for expert in experts:
        if not(expert in memorizer):
            default_weight_update(
                expert,
                metasims, 
                dependencies, 
                str(int(t)+1)
            )

def apriori_assess(
        experts, 
        experts2, 
        weighted_samples, 
        metasims, 
        h, 
        bounds
):
    results = dict()
    for sample in weighted_samples:
        for candidate in sample:
            for expert in experts:
                result = calculate_relatedness_md(expert, candidate[0], candidate[1], metasims, h, 0.2, bounds)
                try:
                    current = results[(expert, candidate[0], candidate[1])]
                    current.append([result, "LiveDOM"])
                    results[(expert, candidate[0], candidate[1])] = current
                except:
                    results[(expert, candidate[0], candidate[1])] = [[result, "LiveDOM"]]
            for expert1 in experts:
                for expert2 in experts:
                    if expert1 != expert2:
                        result2 = calculate_pairwise_md(expert1, expert2, candidate[0], candidate[1],
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
                    result3 = calculate_robustess_md(expert1, expert2, candidate[0], candidate[1],
                     metasims, h, 0.2, bounds)
                    try:
                        current = results[(expert1, candidate[0], candidate[1])]
                        current.append([result3, ("LiveDOR", expert2)])
                        results[(expert1, candidate[0], candidate[1])] = current
                    except:
                        results[(expert1, candidate[0], candidate[1])] = [
                                                                            [result3, ("LiveDOR",expert2)]
                                                                         ]					
    return results

def make_pd_hedge(results, D, t):
    pd_arr = []
    pd_arr_2 = []
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
                    prediction = res[0]
                    density = res[1]
                    pd_arr_2.append([expert, prediction, [prediction, density], res[2], dependency])
                if res_component[1][0] == "LiveDOI":
                    continue
                if res_component[1][0] == "LiveDOR":
                    dependency = ("LiveDOR", res[4], res[5])
                    prediction = res[0]
                    density = res[1]
                    behavior = res[4]	
                    if not(behavior == "Perf"):
                        continue
                    pd_arr_2.append([expert, prediction, [prediction, density], res[2], dependency])
                prediction = mapIntervals(res[0])
                avg += float(prediction) * float(res[1]) * float(ep_state.expert_weight_dic[(res[3], res[2], 
                    dependency, t)]) * D[(expert[2])]
        pd_arr.append([expert, avg])
    return pd_arr, pd_arr_2

def get_real_probs(choices):
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

def calculate_adapted_reward(results, t):
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
                if ((res[2] == "StateTextExtraChars") or (res[2] == "StateTextLengthMetaSimilarity")
                 or (res[2] == "StateSameLingualType") or (res[2] == "StateCandidateWordEmbeddings")):
                    continue
                dependency = None
                if global_dependency == "LiveDOM":
                    dependency = "LiveDOM"
                if global_dependency[0] == "LiveDOI":
                    continue
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
                * float(dor_impact) * float(ep_state.expert_weight_dic[(res[3], res[2], dependency, t)]))			
    return avg