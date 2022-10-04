import numpy as np

from textblob import TextBlob
from datasketch import MinHash
import nltk
import torch
import clip

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_config import *

def handle_metasims(
        point, 
        h, 
        metasims
):
    for metasim in metasims:
        threshold = meta_similarity_thresholds[metasim]
        online_metasim_calculation(
                point,
                metasim, 
                h, 
                threshold
        )

def get_lingual_type(tweet, candidate):
    res = 100000
    tokenized = nltk.word_tokenize(tweet)
    pos = nltk.pos_tag(tokenized)
    blob = TextBlob(tweet)
    nps = blob.noun_phrases
    for np in nps:
        if candidate == np:
            return 0
    for p in pos:
        if p[0] == candidate:
            if p[1] in ep_state.posMapper:
                res = ep_state.posMapper[p[1]]
                return res
            else:
                counter = len(ep_state.posMapper)
                ep_state.posMapper[p[1]] = counter
                return counter
    reses = []
    for p in pos:
        if p[0] in candidate or candidate in p[0]:
            if p[1] in ep_state.posMapper:
                reses.append(ep_state.posMapper[p[1]])
            else:
                counter = len(ep_state.posMapper)
                ep_state.posMapper[p[1]] = counter
                reses.append(counter)
    if len(reses) != 0:
        tempres = ""
        reses.sort()
        gotit = dict()
        for r in reses:
            if not(r in gotit):
                tempres = tempres + str(r)
                gotit[r] = 1
        res = int(tempres)
        return res
    else:
        return res

def online_metasim_calculation(
        c1, 
        typei, 
        h, 
        simi
):
    found = None
    try:
        found = ep_state.global_neighbors[(c1,typei,str(h))]
    except:
        pass
    if found == None:
        h2 = -1
        try:
            h2 = ep_state.candidate_horizon[c1]
        except KeyError:
            pass
        if str(h2) == str(h):
            ep_state.mc += 1
            ep_state.mc1 += 1
            if typei == 'SameLingualType':
                lt = get_lingual_type(ep_state.state_text[c1], ep_state.candidates[c1])
                ep_state.ling_vector[c1] = [float(lt)]
                ep_state.X_ling.append([float(lt)])
                ep_state.Y_ling.append(c1)
            if typei == 'CandidateTextMetaSimilarity':
                d = []
                for i in range(len(ep_state.candidates[c1])):
                    d.append(ep_state.candidates[c1][i])
                m = MinHash(num_perm=128)
                for ch in d:
                    m.update(ch.encode('utf8'))
                ep_state.lsh_1.insert('m' + str(ep_state.mc1), m)
                ep_state.M1[c1] = m	
                indi = 'm' + str(ep_state.mc1)
                ep_state.M_rev_1[indi] = c1
            if typei == 'StateTextExtraChars':
                ep_state.ling_vector[c1] = [float(count_extra_chars(ep_state.state_text[c1]))]
                ep_state.X_extra.append([float(count_extra_chars(ep_state.state_text[c1]))])
                ep_state.Y_extra.append(c1)
            if typei == 'StateSameLingualType':
                lt = []
                lt = get_state_lingual_type(ep_state.state_text[c1])
                d = []
                for p in lt:
                    d.append(p)
                m = MinHash(num_perm=128)
                for ch in d:
                    m.update(ch.encode('utf8'))
                ep_state.lsh_0.insert('m' + str(ep_state.mc), m)
                ep_state.M[c1] = m	
                indi = 'm' + str(ep_state.mc)
                ep_state.M_rev[indi] = c1
            if typei == 'StateTextLengthMetaSimilarity':
                ep_state.ling_vector[c1] = [len(ep_state.state_text[c1])]
                ep_state.X_length.append([len(ep_state.state_text[c1])])
                ep_state.Y_length.append(c1)
        nbrs = None
        X = None
        Y = None
        if typei == "StateTextExtraChars":
            nbrs = ep_state.nbrs_extra
            X = ep_state.X_extra
            Y = ep_state.Y_extra
        if typei == "StateTextLengthMetaSimilarity":
            nbrs = ep_state.nbrs_length
            X = ep_state.X_length
            Y = ep_state.Y_length
        if typei == "SameLingualType":
            nbrs = ep_state.nbrs_ling
            X = ep_state.X_ling
            Y = ep_state.Y_ling
        if (typei == 'SameLingualType' or typei == 'StateTextLengthMetaSimilarity' 
        or typei == 'StateTextExtraChars'):
            nbrs.fit(X)
            h2 = -1
            try:
                h2 = ep_state.candidate_horizon[c1]
            except KeyError:
                pass
            if str(h2) == str(h):
                ind = nbrs.radius_neighbors([ep_state.ling_vector[c1]], simi)
                ep_state.neighbors[c1] = ind
            candiNeighbrs = []
            if not len(ep_state.neighbors[c1][1]) == 0:
                for it in range(len(ep_state.neighbors[c1][1][0])):
                    no = ep_state.neighbors[c1][1][0][it]
                    candiNeighbrs.append(Y[no])
            ep_state.global_neighbors[(c1,typei,str(h))] = candiNeighbrs
        else:
            if typei == "StateSameLingualType":
                h2 = -1
                try:
                    h2 = ep_state.candidate_horizon[c1]
                except KeyError:
                    pass
                if str(h2) == str(h):
                    fin = []
                    nearest = ep_state.lsh_0.query(ep_state.M[c1])
                    for n in nearest:
                        fin.append(ep_state.M_rev[n])
                        sim = ep_state.M[c1].jaccard(ep_state.M[ep_state.M_rev[n]])
                        ep_state.minhash_sim[(c1, ep_state.M_rev[n])] = sim
                        ep_state.minhash_sim[(ep_state.M_rev[n],c1)] = sim
                    ep_state.global_neighbors[(c1,typei,str(h))] = fin
            else:
                if typei == "StateCandidateWordEmbeddings":
                    it = 0
                    candiNeighbrs = []
                    it +=1
                    h2 = -1
                    try:
                        h2 = ep_state.candidate_horizon[c1]
                    except KeyError:
                        pass
                    if str(h2) == str(h):
                        veci1 = None
                        try:
                            veci1 = ep_state.state_vectors[c1]
                        except:
                            veci1 = get_all_vecs(c1, "state")
                        for c2 in ep_state.candidates:
                            if (ep_state.text_id[c1] != ep_state.text_id[c2] and str(ep_state.candidate_horizon[c2]) == str(h)
                             and not("_NIL" in ep_state.candidates[c2])):
                                veci2 = None
                                try:
                                    veci2 = ep_state.state_vectors[c2]
                                except:
                                    veci2 = get_all_vecs(c2, "state")
                                v1_norm = veci1/np.linalg.norm(veci1)
                                v2_norm = veci2/np.linalg.norm(veci2)	
                                res = np.dot(v1_norm, v2_norm)
                                candiNeighbrs.append(c2)
                                ep_state.state_emb_sim[(c1,c2)] = res
                                ep_state.state_emb_sim[(c2,c1)] = res
                        ep_state.global_neighbors[(c1,typei,str(h))] = candiNeighbrs
                else:
                    if typei == "CandidateWordEmbeddings":
                        it = 0
                        candiNeighbrs = []
                        it +=1
                        h2 = -1
                        try:
                            h2 = ep_state.candidate_horizon[c1]
                        except KeyError:
                            pass
                        if str(h2) == str(h):
                            veci1 = None
                            try:
                                veci1 = ep_state.candidate_vectors[c1]
                            except:
                                veci1 = get_all_vecs(c1, "candi")
                            for c2 in ep_state.candidates:
                                if (ep_state.text_id[c1] != ep_state.text_id[c2] and str(ep_state.candidate_horizon[c2]) == str(h)
                                 and not("_NIL" in ep_state.candidates[c2])):
                                    veci2 = None
                                    try:			
                                        veci2 = ep_state.candidate_vectors[c2]
                                    except:
                                        veci2 = get_all_vecs(c2, "candi")
                                    try:
                                        v1_norm = veci1/np.linalg.norm(veci1)
                                        v2_norm = veci2/np.linalg.norm(veci2)	
                                        res = np.dot(v1_norm, v2_norm)
                                        candiNeighbrs.append(c2)
                                        ep_state.embedded_sim[(c1,c2)] = res
                                        ep_state.embedded_sim[(c2,c1)] = res
                                    except:
                                        pass
                            ep_state.global_neighbors[(c1,typei,str(h))] = candiNeighbrs
                    else:
                        if typei == "CandidateTextMetaSimilarity":
                            h2 = -1
                            try:
                                h2 = ep_state.candidate_horizon[c1]
                            except KeyError:
                                pass
                            if str(h2) == str(h):
                                fin = []
                                nearest = ep_state.lsh_1.query(ep_state.M1[c1])
                                for n in nearest:
                                    fin.append(ep_state.M_rev_1[n])
                                    sim = ep_state.M1[c1].jaccard(ep_state.M1[ep_state.M_rev_1[n]])
                                    ep_state.minhash_sim1[(c1, ep_state.M_rev_1[n])] = sim
                                    ep_state.minhash_sim1[(ep_state.M_rev_1[n],c1)] = sim
                                ep_state.global_neighbors[(c1,typei,str(h))] = fin

def count_extra_chars(tweet):
    counthashtext = tweet.count('#')
    countattext = tweet.count("@")	
    return counthashtext + countattext + 1	

def get_state_lingual_type(tweet):
    res = []
    tokenized = nltk.word_tokenize(tweet)
    pos = nltk.pos_tag(tokenized)	
    for p in pos:
        res.append(p[1])
    return res

def init_clip():
    ep_state.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    ep_state.clip_model, _ = clip.load("ViT-B/32", device=ep_state.clip_device)

def get_clip_text_vector(query):
    if ep_state.clip_model is None:
        init_clip()
    text = clip.tokenize(query).to(ep_state.clip_device)      
    with torch.no_grad():
        text_features = ep_state.clip_model.encode_text(text).detach().cpu().numpy().squeeze()
    return text_features
    
def get_all_vecs(candi, typei):
    if typei == "candi":
        candidate_tokens = ep_state.candidates[candi]
        query = [candidate_tokens]
        word_vector = get_clip_text_vector(query) 
        ep_state.candidate_vectors[candi] = word_vector
        return word_vector
    if typei == "state":
        candidate_tokens = ep_state.state_text[candi]
        query = [candidate_tokens]
        word_vector = get_clip_text_vector(query) 
        ep_state.state_vectors[candi] = word_vector
        return word_vector

def get_neighbours():
    for rew in ep_state.rewards:
        temp = []
        try:					
            temp = ep_state.reward_collector[(rew[2],rew[1])]
            if not(rew in temp):
                temp.append(rew)
        except KeyError:
            temp.append(rew)
        ep_state.reward_collector[(rew[2],rew[1])] = temp
    for rew in ep_state.rewards:
        temp = []
        try:					
            temp = ep_state.inv_reward_collector[(rew[2],rew[4])]
            if not(rew in temp):
                temp.append(rew)
        except KeyError:
            temp.append(rew)
        ep_state.inv_reward_collector[(rew[2],rew[4])] = temp