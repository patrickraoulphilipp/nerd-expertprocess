import random
import numpy as np

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_decide import *
from nerd_expertprocess.ep_learn import *
from nerd_expertprocess.ep_metasims import handle_metasims, get_neighbours
from nerd_expertprocess.ep_datasets import format_dataset
from nerd_expertprocess.ep_config import *
from nerd_expertprocess.ep_utils import *

def explore_online_hedge():
    mode = "train"
    ep_state.datapoints = format_dataset(dataset_mode)
    datapoints_set_indexes = np.random.choice(list(ep_state.datapoints.keys()), limit)
    datapoints_set = {}
    for index in datapoints_set_indexes:
        datapoints_set[index] = ep_state.datapoints[index]
    past_points = dict()
    t = 1

    #####init dependencies #####################################################
    dependencies = []
    dependencies.append("LiveDOM")
    for expert in experts_dic["0"]:
        ep_state.expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
        for b in behav["DOI"]:
            dependencies.append(("LiveDOI", b, expert))
    for expert in experts_dic["1"]:
        ep_state.expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
        for b in behav["DOI"]:
            dependencies.append(("LiveDOI", b, expert))
    for expert in experts_dic["1"]:
        for b in behav["DOR"]:
            dependencies.append(("LiveDOR", b, expert))
   
    init_weights(experts_dic['0'], metasims_dic['0.1'], dependencies, str(t))
    init_weights(experts_dic['1'], metasims_dic['1'], dependencies, str(t))
   
    #####start learning #######################################################
    for pont_no, point in enumerate(datapoints_set):
        print(
                't={}/{} - precision={}; recall={}; failed points={}; failed_experts={}'
                .format(
                    pont_no, 
                    len(datapoints_set), 
                    get_precision(TRAIN_WINDOW), 
                    get_recall(TRAIN_WINDOW),
                    ep_state.failed_points,
                    ep_state.failed_experts
                )
        )
        ep_state.tweetid[point] = point
        text = ep_state.datapoints[point][0]
        ep_state.text_id[point] = point
        ep_state.candidate_horizon[point] = "0"
        ep_state.candidates[point] = text
        ep_state.state_text[point] = text
        weighted_samples = dict()
        weighted_samples['0'] = [[[point, point]]]
        fail = False
        for h in H:
            for ws in weighted_samples[h]:
                for candi in ws:
                    handle_metasims(candi[1], h, metasims_dic[h])
            get_neighbours()
            results1 = dict()
            if h == "0":
                results1 = apriori_assess(
                               experts_dic[h], 
                               experts_dic[str(int(h)+1)], 
                               weighted_samples[h], 
                               metasims_dic[h], 
                               h, 
                               bounds
                            )
            else:
                results1 = apriori_assess(
                               experts_dic[h], 
                               [], 
                               weighted_samples[h], 
                               metasims_dic[h], 
                               h, 
                               bounds
                            )
            candis, chosen_experts = choose_and_execute(budget[h], results1, h, str(t)) 
            if len(chosen_experts) == 0:
                ep_state.failed_points += 1
                fail = True
            if h == "0":
                h = "0.1"
                for ca in candis:
                    handle_metasims(ca[1],h, metasims_dic[h])
            get_neighbours()
            results2 = dict()
            us = None
            pds = None
            if h in ['0', '0.1']:
                temph = "0"
                results2, us, pds = deal_with_weights_preselection(
                                        candis, 
                                        experts_dic[h], 
                                        experts_dic[str(int(temph)+1)],
                                        metasims_dic[h],
                                        str(t), 
                                        bounds, 
                                        h, 
                                        md_threshold, 
                                    )
            else:
                results2, us, pds = deal_with_weights_preselection(
                                        candis,
                                        experts_dic[h], 
                                        [], 
                                        metasims_dic[h], 
                                        str(t), 
                                        bounds, 
                                        h, 
                                        md_threshold,
                                    )
            newh = -1
            if float(h) == 0.1:
                newh = 1
            else:
                newh = int(h) + 1
            weighted_samples[str(newh)] = sample_candidates(
                                                h, 
                                                point, 
                                                str(t), 
                                                results2, 
                                                fail
                                          )

            deal_with_weights_postselection(
                    us, 
                    pds, 
                    t, 
                    metasims_dic[h], 
                    mode, 
                    dependencies, 
                    experts_dic[h], 
            )
        past_points[point] = 1	
        if len(past_points) == limit:
            break
        if len(past_points) == lower_bound:
            mode = "test"	
            print("starting to test")
            reset_statistics()
        t += 1

def reset_statistics():
    ep_state.Y_Pred = []
    ep_state.Y_Label = []
    ep_state.Y_Label_text = []
    ep_state.Y_Pred_count = 0
    ep_state.Y_Preds = []
