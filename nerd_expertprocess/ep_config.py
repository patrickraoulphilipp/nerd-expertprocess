STASH_PATH = '/PATH/TO/STASH/'
MICROPOST_PATH = './data/micro_14.txt'
SPOTLIGHT_PATH = './data/spotlight.n3'

lower_bound = 200
limit = 230
budget = dict()
budget['0'] = 2
budget["0.1"] = 2
budget["1"] = 4
dataset_mode = "Microposts"

H = ['0', '1']
surrogate_H = ['0', '0.1', '1']
p_min_global = 0.1
global_min_prob_hedge = 0.08
md_threshold = 0.2
TRAIN_WINDOW = 30
default_confidence = 0.5

behav = dict()
behav["DOI"] = ["Agree", "Perf", "ConfPerf", "IndError"]
behav["DOR"] = ["Perf", "ConfPerf", "IndError"]
behav["DOI"] = ["Agree", "Perf", "IndError"]
behav["DOR"] = ["Perf", "IndError"]

experts_dic = dict()
experts_dic["0"] = ["01", "02", "03"]
experts_dic["0.1"] = ["01", "02", "03"]
experts_dic["1"] = ["11", "12", "13"]

metasims_dic = dict()
metasims_dic["0"] = ['StateCandidateWordEmbeddings']
metasims_dic["0.1"] =  ['SameLingualType', 'CandidateWordEmbeddings']
metasims_dic["1"] = ['SameLingualType', 'CandidateWordEmbeddings']
# activate more metadependencies like this:
# metasims_dic["0"] = ['StateTextLengthMetaSimilarity', 'StateCandidateWordEmbeddings', 'StateSameLingualType', 'StateTextExtraChars']
# metasims_dic["0.1"] =  ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']
# metasims_dic["1"] = ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']

meta_similarity_thresholds = {
    'SameLingualType' : 0.5,
    'CandidateTextMetaSimilarity' : None,
    'CandidateWordEmbeddings' : None,
    'StateTextLengthMetaSimilarity' : 40,
    'StateCandidateWordEmbeddings' : None,
    'StateSameLingualType' : 4,
    'StateTextExtraChars' : 2
}

min_samples_bound = dict()
min_samples_bound['SameLingualType'] = 16.0 
min_samples_bound['CandidateWordEmbeddings'] = 35.0
min_samples_bound['CandidateTextMetaSimilarity'] = 35.0
min_samples_bound['StateTextLengthMetaSimilarity'] = 5.0 
min_samples_bound['StateCandidateWordEmbeddings'] = 40.0
min_samples_bound['StateSameLingualType'] = 10.0
min_samples_bound['StateTextExtraChars'] = 1.

bounds = dict()
for h in surrogate_H:
    for metasim in metasims_dic[h]:
        bounds[(metasim, h)] = lower_bound / min_samples_bound[metasim]