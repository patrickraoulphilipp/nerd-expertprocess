from __future__ import print_function
from SPARQLWrapper import SPARQLWrapper, JSON
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
from ep_util import *
import os, shutil
import ast
import subprocess
import commands
import operator
import difflib
import random
import unicodedata
import time
import datetime
import math
import nltk
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

class EP(object):

	def __init__(self, limit=None, lower_bound=None, budget=None, dataset_mode=None, options=None):
		self.limit = limit
		self.posMapper = dict()
		self.budget = budget
		self.bounds = dict()
		self.getRewards = dict()
		self.getRewardsBy = dict()
		self.IndependenceVectors = []
		self.RelatednessVectors = []
		self.RobustnessVectors = []
		self.candidateVectors = dict()
		self.stateVectors = dict()

		self.fittedParameterizedModel = dict()
		self.candidates = dict()
		self.candidateBeliefs = dict()
		self.textid = dict() 
		self.stateText = dict()
		self.stateType = dict()
		self.candidateBeliefsArr = []
		self.beginCandidates = dict()
		self.endCandidates = dict()
		self.text = dict()
		self.texttypes = dict()
		self.annotatedStateText = dict()
		self.hasReward = dict()
		self.tweetid = dict()
		self.textWithEntites = dict()
		self.hasState = dict()
		self.hypos = []
		self.rewards = []
		self.executed = dict()
		self.datapoints = dict()
		self.candidateHorizon = dict()
		self.horizon = dict()
		self.candidatesBy = dict()
		self.allrewards = dict()
		self.stateCandis = dict()

		self.state_dic = dict()
		self.candidateRewards = dict()
		self.stateVectors = dict()
		self.experts_dic = dict()
		self.dependencies = []
		self.globalNeighbors = dict()
		self.minhashsim = dict()
		self.minhashsim1 = dict()
		self.stateEmbSim = dict()
		self.embeddedsim = dict()
		self.neighbors = dict()
		self.vector = dict()
		self.X_length = []
		self.Y_length = []
		self.X_extra = []
		self.Y_extra = []
		self.X_ling = []
		self.Y_ling = []
		self.Mrev = dict()
		self.M = dict()
		self.Mrev1 = dict()
		self.M1 = dict()
		self.nbrs_length = NearestNeighbors(radius=30)
		self.nbrs_extra = NearestNeighbors(radius=2)
		self.nbrs_ling = NearestNeighbors(radius=0)
		self.mc = 0
		self.mc1 = 1
		self.lsh0 = MinHashLSH(threshold=0.2, num_perm=128)
		self.lsh1 = MinHashLSH(threshold=0.2, num_perm=128)
		self.expert_weight_dic = dict()
		self.p_min = 0.05

		self.Y_theo = dict()
		self.Y_theo["1"] = set()
		self.Y_theo["0.1"] = set()
		self.Y_theo["0"] = set()

		self.globalT = None
		self.globalmode = None
		self.lower_bound = lower_bound
		self.dataset_mode = dataset_mode

		self.experts_dic = dict()
		self.experts_dic["0"] = ["01", "02", "03"]
		self.experts_dic["0.1"] = ["01", "02", "03"]
		self.experts_dic["1"] = ["11", "12", "13"]
		self.metasims_dic = dict()
		self.metasims_dic["0"] = ['StateTextLengthMetaSimilarity', 'StateCandidateWordEmbeddings', 'StateSameLingualType', 'StateTextExtraChars']
		self.metasims_dic["0.1"] =  ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']
		self.metasims_dic["1"] = ['SameLingualType', 'CandidateWordEmbeddings', 'CandidateTextMetaSimilarity']
		self.bounds = dict()
		self.bounds[('SameLingualType', "0")] = 20
		self.bounds[('CandidateWordEmbeddings', "0")] = 20
		self.bounds[('CandidateTextMetaSimilarity', "0")] = 20
		self.bounds[('StateTextLengthMetaSimilarity', "0")] = 20
		self.bounds[('StateCandidateWordEmbeddings', "0")] = 20
		self.bounds[('StateSameLingualType', "0")] = 20
		self.bounds[('StateTextExtraChars', "0")] = 20
		self.bounds[('SameLingualType', "1")] = 20
		self.bounds[('SameLingualType', "0.1")] = 20
		self.bounds[('CandidateWordEmbeddings', "0.1")] = 20
		self.bounds[('CandidateTextMetaSimilarity', "0.1")] = 20
		self.bounds[('StateTextLengthMetaSimilarity', "0.1")] = 20
		self.bounds[('StateCandidateWordEmbeddings', "0.1")] = 20
		self.bounds[('StateSameLingualType', "0.1")] = 20
		self.bounds[('StateTextExtraChars', "0.1")] = 20
		self.bounds[('SameLingualType', "1")] = 20
		self.bounds[('CandidateWordEmbeddings', "1")] = 20
		self.bounds[('CandidateTextMetaSimilarity', "1")] = 20
		self.bounds[('StateTextLengthMetaSimilarity', "1")] = 20
		self.bounds[('StateCandidateWordEmbeddings', "1")] = 20
		self.bounds[('StateSameLingualType', "1")] = 20
		self.bounds[('StateTextExtraChars', "1")] = 20
		self.behav = dict()
		self.behav["DOI"] = ["Agree", "Perf", "ConfPerf", "IndError"]
		self.behav["DOR"] = ["Perf", "ConfPerf", "IndError"]
		self.behav["DOI"] = ["Agree", "Perf", "IndError"]
		self.behav["DOR"] = ["Perf", "IndError"]
		logging.getLogger("urllib3").setLevel(logging.WARNING)
		try:
			opts = process_args(options)
			self.wv = wvlib.load(opts.vectors, max_rank=opts.max_rank)
	     	  	self.wv = self.wv.normalize()
		except Exception, e:
			print >> sys.stderr, 'Error: %s' % str(e)

	def getAllVecs(self, candi, typei):
		if typei == "candi":
			query = [[self.candidates[candi]]]
			words = [w for q in query for w in q]
			veci1 = None
			try: 
				veci1 = self.wv.words_to_vector(words)
			except:
				veci1 = getTweetVector(self.candidates[candi],1)
			if veci1 != None:
				self.candidateVectors[candi] = veci1
			return veci1
		if typei == "state":
			veci2 = None
			veci2 = getTweetVector(self.candidates[candi],1)
			if veci2 != None:
				self.stateVectors[candi] = veci2
			return veci2	

	def doMS(self, point, h, metasims):
		if 'SameLingualType' in metasims:
			self.onlineMetasimcomp(point, 'SameLingualType', h, 0.5)
		if 'CandidateTextMetaSimilarity' in metasims:
			self.onlineMetasimcomp(point, 'CandidateTextMetaSimilarity', h, None)
		if 'CandidateWordEmbeddings' in metasims:
			self.onlineMetasimcomp(point, 'CandidateWordEmbeddings', h, None)
		if 'StateTextLengthMetaSimilarity' in metasims:
			self.onlineMetasimcomp(point, 'StateTextLengthMetaSimilarity', h, 40)
		if 'StateCandidateWordEmbeddings' in metasims:
			self.onlineMetasimcomp(point, 'StateCandidateWordEmbeddings', h, None)
		if 'StateSameLingualType' in metasims:
			self.onlineMetasimcomp(point, 'StateSameLingualType', h, 4)
		if 'StateTextExtraChars' in metasims:
			self.onlineMetasimcomp(point, 'StateTextExtraChars', h, 2) 

	def onlineMetasimcomp(self, c1, typei, h, simi):
		found = None
		try:
			found = self.globalNeighbors[(c1,typei,str(h))]
		except:
			pass
		if found == None:
			h2 = -1
			try:
				h2 = self.candidateHorizon[c1]
			except KeyError:
				pass
			if str(h2) == str(h):
				self.mc += 1
				self.mc1 += 1
				if typei == 'SameLingualType':
					lt = -1
					try:
						lt = getLingualType(self.stateText[c1], self.candidates[c1])
					except:
						pass
					self.vector[c1] = [float(lt)]
					self.X_ling.append([float(lt)])
					self.Y_ling.append(c1)
				if typei == 'CandidateTextMetaSimilarity':
					d = []
					for i in range(len(self.candidates[c1])):
						d.append(self.candidates[c1][i])
					m = MinHash(num_perm=128)
					for ch in d:
						m.update(ch.encode('utf8'))
					self.lsh1.insert('m' + str(self.mc1), m)
					self.M1[c1] = m	
					indi = 'm' + str(self.mc1)
					self.Mrev1[indi] = c1
				if typei == 'StateTextExtraChars':
					self.vector[c1] = [float(countExtras(self.stateText[c1]))]
					self.X_extra.append([float(countExtras(self.stateText[c1]))])
					self.Y_extra.append(c1)
				if typei == 'StateSameLingualType':
					lt = []
					lt = getStateLingualType(self.stateText[c1])
					d = []
					for p in lt:
						d.append(p)
					m = MinHash(num_perm=128)
					for ch in d:
						m.update(ch.encode('utf8'))
					self.lsh0.insert('m' + str(self.mc), m)
					self.M[c1] = m	
					indi = 'm' + str(self.mc)
					self.Mrev[indi] = c1
				if typei == 'StateTextLengthMetaSimilarity':
					self.vector[c1] = [len(self.stateText[c1])]
					self.X_length.append([len(self.stateText[c1])])
					self.Y_length.append(c1)
			nbrs = None
			X = None
			Y = None
			if typei == "StateTextExtraChars":
				nbrs = self.nbrs_extra
				X = self.X_extra
				Y = self.Y_extra
			if typei == "StateTextLengthMetaSimilarity":
				nbrs = self.nbrs_length
				X = self.X_length
				Y = self.Y_length
			if typei == "SameLingualType":
				nbrs = self.nbrs_ling
				X = self.X_ling
				Y = self.Y_ling
			if (typei == 'SameLingualType' or typei == 'StateTextLengthMetaSimilarity' 
			or typei == 'StateTextExtraChars'):
				nbrs.fit(X)
				h2 = -1
				try:
					h2 = self.candidateHorizon[c1]
				except KeyError:
					pass
				if str(h2) == str(h):
					ind = nbrs.radius_neighbors([self.vector[c1]], simi)
					self.neighbors[c1] = ind
				candiNeighbrs = []
				if not len(self.neighbors[c1][1]) == 0:
					for it in range(len(self.neighbors[c1][1][0])):
						no = self.neighbors[c1][1][0][it]
						candiNeighbrs.append(Y[no])
				self.globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
			else:
				if typei == "StateSameLingualType":
					h2 = -1
					try:
						h2 = self.candidateHorizon[c1]
					except KeyError:
						pass
					if str(h2) == str(h):
						fin = []
						nearest = self.lsh0.query(self.M[c1])
						for n in nearest:
							fin.append(self.Mrev[n])
							sim = self.M[c1].jaccard(self.M[self.Mrev[n]])
							self.minhashsim[(c1,self.Mrev[n])] = sim
							self.minhashsim[(self.Mrev[n],c1)] = sim
						self.globalNeighbors[(c1,typei,str(h))] = fin
				else:
					if typei == "StateCandidateWordEmbeddings":
						it = 0
						candiNeighbrs = []
						it +=1
						h2 = -1
						try:
							h2 = self.candidateHorizon[c1]
						except KeyError:
							pass
						if str(h2) == str(h):
							veci1 = None
							try:
								veci1 = stateVectors[c1]
							except:
								veci1 = self.getAllVecs(c1, "state")
							for c2 in self.candidates:
								if (self.textid[c1] != self.textid[c2] and str(self.candidateHorizon[c2]) == str(h)
								 and not("_NIL" in self.candidates[c2])):
									veci2 = None
									try:
										veci2 = stateVectors[c2]
									except:
										veco2 = self.getAllVecs(c2, "state")
									try:
										v1_norm = veci1/np.linalg.norm(veci1)
										v2_norm = veci2/np.linalg.norm(veci2)	
										numba = np.dot(v1_norm, v2_norm)
										candiNeighbrs.append(c2)
										self.stateEmbSim[(c1,c2)] = numba
										self.stateEmbSim[(c2,c1)] = numba
									except:
										pass
							self.globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
					else:
						if typei == "CandidateWordEmbeddings":
							it = 0
							candiNeighbrs = []
							it +=1
							h2 = -1
							try:
								h2 = self.candidateHorizon[c1]
							except KeyError:
								pass
							if str(h2) == str(h):
								veci1 = None
								try:
									veci1 = self.candidateVectors[c1]
								except:
									veci1 = self.getAllVecs(c1, "candi")
								for c2 in self.candidates:
									if (self.textid[c1] != self.textid[c2] and str(self.candidateHorizon[c2]) == str(h)
									 and not("_NIL" in self.candidates[c2])):
										veci2 = None
										try:			
											veci2 = self.candidateVectors[c2]
										except:
											veci2 = self.getAllVecs(c2, "candi")
										try:
											v1_norm = veci1/np.linalg.norm(veci1)
											v2_norm = veci2/np.linalg.norm(veci2)	
											res = np.dot(v1_norm, v2_norm)
											candiNeighbrs.append(c2)
											self.embeddedsim[(c1,c2)] = res
											self.embeddedsim[(c2,c1)] = res
										except:
											pass
								self.globalNeighbors[(c1,typei,str(h))] = candiNeighbrs
						else:
							if typei == "CandidateTextMetaSimilarity":
								h2 = -1
								try:
									h2 = self.candidateHorizon[c1]
								except KeyError:
									pass
								if str(h2) == str(h):
									fin = []
									nearest = self.lsh1.query(self.M1[c1])
									for n in nearest:
										fin.append(self.Mrev1[n])
										sim = self.M1[c1].jaccard(self.M1[self.Mrev1[n]])
										self.minhashsim1[(c1,self.Mrev1[n])] = sim
										self.minhashsim1[(self.Mrev1[n],c1)] = sim
									self.globalNeighbors[(c1,typei,str(h))] = fin
			hor = -1
			try:
				hor = self.candidateHorizon[c1]
			except KeyError:
				hor = 5
			if str(hor) == str(h):
				kkk = []
				avg = 0.0
				newNeighbs = []
				for c2 in self.globalNeighbors[(c1,typei,str(h))]:
					kkk.append(c2)
					if typei == 'SameLingualType':
						newNeighbs.append([c2, str(1.0)])
					if typei == 'CandidateTextMetaSimilarity':
						simi = self.minhashsim1[(c1,c2)]
						newNeighbs.append([c2, simi])
					if typei == 'CandidateWordEmbeddings':
						simi = self.embeddedsim[(c1,c2)]
						newNeighbs.append([c2, simi])
					if typei == 'StateTextExtraChars':
						numba = 0
						numba = float(countExtras(self.stateText[c1])) / float(countExtras(self.stateText[c2]))
						if numba > 1:
							numba = 1 / float(numba)
						newNeighbs.append([c2, str(numba)])
					if typei == 'StateSameLingualType':
						simi = self.minhashsim[(c1,c2)]
						newNeighbs.append([c2, simi])
					if typei == 'StateCandidateWordEmbeddings':
						simi = self.stateEmbSim[(c1,c2)]
						newNeighbs.append([c2, simi])
					if typei == 'StateTextLengthMetaSimilarity':
						numba = 0
						numba = float(len(self.stateText[c1])) / float(len(self.stateText[c2]))
						if numba > 1:
							numba = 1 / float(numba)
						newNeighbs.append([c2,str(numba)]) 
				self.globalNeighbors[(c1,typei,str(h))] = newNeighbs

	def getNeigbs(self):
		for rew in self.rewards:
			temp = []
			try:					
				temp = self.getRewards[(rew[2],rew[1])]
				if not(rew in temp):
					temp.append(rew)
			except KeyError:
				temp.append(rew)
			self.getRewards[(rew[2],rew[1])] = temp
		for rew in self.rewards:
			temp = []
			try:					
				temp = self.getRewardsBy[(rew[2],rew[4])]
				if not(rew in temp):
					temp.append(rew)
			except KeyError:
				temp.append(rew)
			self.getRewardsBy[(rew[2],rew[4])] = temp

	def newLivedoM(self, protagonist, sid, cid, metasims, h, threshold, bounds):
		tweedid = self.textid[cid]
		is_nil = False
		is_av = False
		metasimarray = []
		flag = 0
		RelatednessVector = None
		gotcha = None
		RelatednessVector = [protagonist,sid,cid,[],self.globalmode,self.globalT]
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
				neighs = self.globalNeighbors[(cid,metasim,str(h))]
			except:
				pass
			for neigh in neighs:
				rews2 = []
				try:
					rews2 = self.getRewards[(protagonist,neigh[0])]
				except KeyError:
					continue
				for rew2 in rews2:
					is_nil_sim = False
					try:
						if "_NIL" in self.candidates[rew2[4]]:
					 		is_nil_sim = True
					except:
						is_av = False
					if (rew2[2] == protagonist and sid != rew2[0] and cid != rew2[1] 
					and self.tweetid != rew2[7]):
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
		self.RelatednessVectors.append(RelatednessVector)
		return metasimarray

	def newLivedoIPairwise(self, protagonist, competitor, sid, cid, metasims, h, threshold, bounds):	
		predictors = []
		for metasim in metasims:
			agreementRate = []
			carr = []
			performanceRate = []
			confidenceAdjustedPerformanceRate = []
			independentErrorRate = []
			computeM = 0
			ccounter = 0
			try:
				bla =  self.globalNeighbors[(cid,metasim,str(h))]
			except:
				self.globalNeighbors[(cid,metasim,str(h))] = []
			for neigh in self.globalNeighbors[(cid,metasim,str(h))]:	
				rewhelpers = []
				try:
					rewhelpers = self.getRewards[(protagonist,neigh[0])]
				except KeyError:
					continue
				for rewhelper in rewhelpers: 
					if rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1
					] and self.tweetid != rewhelper[7]:
						similar = neigh[1]	
						if similar >= threshold:
							rews2 = []
							try:
								rews2 = self.getRewards[(competitor,neigh[0])]
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
			self.IndependenceVectors.append([[protagonist, competitor], sid, cid, metasim, [[agreeresult, "Agree", density]
				, [(1.0 - float(agreeresult)) , "NegConfAgree", density], [confagreeresult, "ConfAgree", density], 
				[confperfresult, "ConfPerf", density], [perfresult, "Perf", density], [indresult, "Error", density]],
				 self.globalmode, self.globalT])
			predictors.extend([[agreeresult, density, metasim, protagonist, "Agree", competitor], [confagreeresult,
			density, metasim, protagonist, "ConfAgree", competitor], [confperfresult, density, metasim, protagonist,
			"ConfPerf", competitor], [perfresult, density, metasim, protagonist, "Perf", competitor], [indresult,
			density, metasim, protagonist, "IndError", competitor]])
		return predictors

	def newLivedoR(self, protagonist, competitor, sid, cid, metasims, h, threshold, bounds):
		threshold = 0.5
		predictors = []
		for metasim in metasims:
			performanceRate = []
			confidenceAdjustedPerformanceRate = []
			independentErrorRate = []
			computeM = 0
			ccounter = 0
			for neigh in self.globalNeighbors[(cid,metasim,str(h))]:
				rewhelpers = []
				try:
					rewhelpers = self.getRewards[(protagonist,neigh[0])]
				except KeyError:
					continue
				for rewhelper in rewhelpers:
					if (rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1] 
					and self.tweetid != rewhelper[7] and rewhelper[0] != rewhelper[1]):
						similar = float(neigh[1])
						if similar >= threshold:
							computeM += float(similar)
							ccounter += 1
							rews2 = []	
							try:
								rews2 = self.getRewards[(competitor,rewhelper[4])]
							except KeyError:
								try:
									if "_NIL" in self.candidates[rewhelper[4]] and rewhelper[6] < 0.5:
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
			self.RobustnessVectors.append([[protagonist, competitor],sid, cid, metasim, 
				[[confperfresult, "ConfPerf", density], [perfresult, "Perf", density],
				 [indresult, "Error", density]],self.globalT])
			predictors.extend([ [indresult, density, metasim, protagonist, "IndError", competitor],
			 [perfresult, density, metasim, protagonist, "Perf", competitor], 
			 [confperfresult, density, metasim, protagonist, "ConfPerf", competitor]])
		return predictors

	def initWeights(self, experts, metasims, dependencies, t):
		for expert in experts:
			self.initWeight(expert, metasims, dependencies, t)

	def initWeight(self, expert, metasims, dependencies, t):
		for ms in metasims:
			for dependency in dependencies:
				self.expert_weight_dic[(expert, ms, dependency, t)] = 1.0

	def defaultUpdateWeight(self, expert, metasims, dependencies, t):
		for ms in metasims:
			for dependency in dependencies:
				self.expert_weight_dic[(expert, ms, dependency, t)] = self.expert_weight_dic[(expert, ms, dependency, str(int(t)-1))] 

	def dealWithWeights_pre(self, weighted_samples, experts, experts2, metasims, t, bounds, h, threshold, mode, 
		dependencies, chosen_experts):
		self.globalmode = "post"
		print("Dealing w/ weights H=" +str(h))
		uniqueStates = set()
		memorizer = set()
		results = dict()
		D = dict()
		give_back = []
		for candidate_state in weighted_samples: 
			D[(candidate_state[0], candidate_state[1])] = 1.0
			for expert in experts:
				result = self.newLivedoM(expert, candidate_state[0], candidate_state[1], metasims, h, 
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
						result2 = self.newLivedoIPairwise(expert1, expert2, candidate_state[0], 
							candidate_state[1], metasims, h, threshold, bounds)
						try:
							current = results[(expert1, candidate_state[0], candidate_state[1])]
							current.append([result2, ("LiveDOI", expert2)])
							results[(expert1, candidate_state[0], candidate_state[1])] = current
						except:
							results[(expert1, candidate_state[0], candidate_state[1])] = [[result2, 
							("LiveDOI", expert2)]]
				for expert2 in experts2:
					result3 = self.newLivedoR(expert1, expert2, candidate_state[0], candidate_state[1],
					 metasims, h, threshold, bounds)
					try:
						current = results[(expert1, candidate_state[0], candidate_state[1])]
						current.append([result3, ("LiveDOR", expert2)])
						results[(expert1, candidate_state[0], candidate_state[1])] = current
					except:
						results[(expert1, candidate_state[0], candidate_state[1])] = [[result3,
						 ("LiveDOR", expert2)]]
		if h != "0":
			pd, pd2 = self.makePD_hedge(results, D, t, chosen_experts)
			real_pd = getRealProbs(pd)
			pds = [real_pd,pd2]
			total = sum(w for c, w in pd)
			for us in uniqueStates:
				allRews = []
				try:
					allRews = self.allrewards[us]
				except:
					continue
				for rew in allRews:
					if rew[0] == rew[1]:
						continue
					if mode == "batch":
						self.updateWeights(rew[2], total, rew[0], rew[1], rew[6], pds, metasims, t, mode,
						 dependencies, False)
					memorizer.add(rew[2])
					give_back.append([rew, results[rew[2], rew[0], rew[1]]])
			print("HAVE TO DEF UPDATE WEIGHTS!!!!!!!!")
			if mode == "batch":
				for expert in experts:
					if not(expert in memorizer):
						print("Expert " +str(expert) + " not chosen!!!")
						self.defaultUpdateWeight(expert, metasims, dependencies, str(int(t)+1))
		return give_back, uniqueStates, pds, results, D

	def updateWeights(self, expert, total, state, candi, retrieved_reward, pds, metasims, t, mode, dependencies, bonus):
		pd, pd2 = pds
		if float(total) == 0.0:
			total = 0.1
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
						updated_weight = self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
					except:
						self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = self.expert_weight_dic[(expert, ms, dependency, str(int(t)))]
				else:
					reward = ( 1 - abs( float(retrieved_reward) - float(prediction)  ))			
					reward = mapIntervals(reward)
					reward = float(reward) * float(density)
					p_expert += p_min_global
					reward = float(reward) / float(p_expert) 
					try:
						updated_weight = self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))]
					except:
						self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = self.expert_weight_dic[(expert, ms, dependency, str(int(t)))]
					if mode == "batch":
						self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = self.expert_weight_dic[(expert, ms, dependency, str(t))]
					else:
						try:	
							update = math.exp(reward * p_min_global)
							self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = float(self.expert_weight_dic[(expert, ms, dependency, str(t))]) * float(math.exp(reward * p_min_global))
						except:	
							update = 0.0
						 	try:
								update = math.exp(reward * p_min)
								self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = self.expert_weight_dic[(expert, ms, dependency, str(t))]						
							except:
								self.expert_weight_dic[(expert, ms, dependency, str(int(t)+1))] = math.exp(100)

	def getCRs(self):
		for rew in self.rewards:
			self.candidateRewards[rew[4]] = rew[6]

	def dealWithWeights_post(self, uniqueStates, pds, t, metasims, mode, dependencies, experts, winna,
	 chosen_experts):
		memorizer = set()
		pd, pd2 = pds
		total = sum(w for c, w in pd)
		if mode == "train":
			for us in uniqueStates:
				collector = dict()
				allRews = []
				try:
					allRews = self.allrewards[us]
				except:
					continue
				for rew in allRews:
					bonus = False
					if rew[0] == rew[1]:
						continue
					if (not(rew[4]) in winna and self.candidateRewards[rew[4]] >= 0.5) or (rew[4] in 
						winna and self.candidateRewards[rew[4]] < 0.5):
						bonus = True
					print("Updated weights for " +str(rew[2]))
					self.updateWeights(rew[2], total, rew[0], rew[1], rew[6], pds, metasims, t, mode,
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
						pairs = self.datapoints[rew[7]][3]
						count = 0
						for rew in rews:
							count += rew[6]
						diff = len(pairs) - rew[6]
		else:
			for us in uniqueStates:
				allRews = []
				try:
					allRews = self.allrewards[us]
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
				self.defaultUpdateWeight(expert, metasims, dependencies, str(int(t)+1))

	def apriori_assess(self, experts, experts2, weighted_samples, metasims, h, bounds):
		self.globalmode = "pre"
		pds = []
		results = dict()
		for sample in weighted_samples:
			for candidate in sample:
				for expert in experts:
					result = self.newLivedoM(expert, candidate[0], candidate[1], metasims, h, 0.2, bounds)
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
							result2 = self.newLivedoIPairwise(expert1, expert2, candidate[0], candidate[1],
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
						result3 = self.newLivedoR(expert1, expert2, candidate[0], candidate[1],
						 metasims, h, 0.2, bounds)
						try:
							current = results[(expert1, candidate[0], candidate[1])]
							current.append([result3, ("LiveDOR", expert2)])
							results[(expert1, candidate[0], candidate[1])] = current
						except:
							results[(expert1, candidate[0], candidate[1])] = [[result3, ("LiveDOR",
							 expert2)]]					
		return results

	def choose_and_execute(self, budget, weights, h, t, metasims):
		self.D_used = []
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
				pd, pd2 = self.makePD_hedge(weights, D, t, chosen_experts)
				chosen = weighted_choice_hedge(pd)
				real_probs = getRealProbs(pd)
				self.D_used.append([D, chosen])
				temp_candis = self.execute(chosen[0], h)
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

	def makePD_hedge(self, results, D, t, chosen_experts):
		print("CURRENT T=" + str(t))
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
					avg += float(prediction) * float(res[1]) * float(self.expert_weight_dic[(res[3], res[2], 
						dependency, t)]) * D[(expert[2])]
			pd_arr.append([expert, avg])
		return pd_arr, pd_arr_2

	def getNewSamples(self, metasims, h, samplesize, threshold, currentTweetID, t, bounds, results, mode):
		self.getCRs()
		reward_belief_collector_KL = [] 
		some_collector = dict()
		reward_collector = dict()
		uniqueStates = set()
		by_exp = dict()
		for result in results:
			rew = result[0]
			regressors = result[1]
			print("(" +str(rew[0]) + ") Thinking about: " +str(self.candidates[rew[1]]) 
				+ " saying: " +str(self.candidates[rew[4]]) +" by: " +str(rew[2]))
			if not("_NIL" in self.candidates[rew[4]]):
				self.Y_theo[str(h)].add((rew[4], self.candidates[rew[4]], rew[6]))
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
		denom = self.getDenom(all_best, t)
		for best in reward_collector:
			rew = self.computeItSimple(reward_collector[best], t, 0.0, "not_test", denom)
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
			r2 = range(int(self.beginCandidates[p]), int(self.endCandidates[p]))
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
							if begin > int(self.beginCandidates[part[1]]):
								begin = int(self.beginCandidates[part[1]])
							if end < int(self.endCandidates[part[1]]):
								end = int(self.endCandidates[part[1]])						
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
							range1 = range(int(self.beginCandidates[prior[0]]), 
								int(self.endCandidates[prior[0]])+1)
						except KeyError:
							continue
						try:
							range2 = range(int(self.beginCandidates[proposition[0]]), 
								int(self.endCandidates[proposition[0]])+1)
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
					print("-> " +self.candidates[ch[0]] + ") " +str(by_exp[ch[0]]))
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
					self.Y_Pred_global.append(0)
				else:
					self.Y_Pred_global.append(1)
				res = round(self.candidateRewards[possi])
				self.Y_Label_global.append(res)
			if h == "1":
				self.Y_Label_count += len(self.datapoints[currentTweetID][3])
				self.Y_Labels.extend(self.datapoints[self.tweetid[currentTweetID]][3])
				for part in mychoice:
					try:
						print("-> " +self.candidates[part] + " (" +str(self.candidateRewards[part])
						 + ") " +str(by_exp[part]))
					except:
						pass
					if not("_NIL" in self.candidates[part]):
						self.Y_Pred.append(1.0)
						self.Y_Label.append(self.candidateRewards[part])
						self.Y_Label_text.append(self.candidates[part])
						self.Y_Pred_count += int(self.candidateRewards[part])
						self.Y_Preds.append(self.candidates[part])
			else:
				self.Y_Label_count_1 += len(self.datapoints[currentTweetID][3])
				for part in mychoice:
					print("-> " +self.candidates[part] + " (" +str(self.candidateRewards[part]) 
						+ ")")
					try:
						self.chosen_exps.append(by_exp[part])
					except:
						self.chosen_exps.append("Add_NIL")
					if not("_NIL" in self.candidates[part]):
						self.Y_Pred_1.append(1.0) #TODO just use "1"?!
						self.Y_Label_1.append(self.candidateRewards[part])
						self.Y_Label_text_1.append(self.candidates[part])
						self.Y_Pred_count_1 += int(self.candidateRewards[part])
		final_combis = []
		combiIDcounter = 0	
		state_choices = []
		for combi in [mychoice]:
			if len(combi) == 0:
				continue
			newStateID = currentTweetID + str(combiIDcounter)
			newText = annotateCandidatesInText(self.candidates, self.beginCandidates, self.endCandidates, 
				combi, self.stateText[currentTweetID])		
			self.textWithEntites[newStateID] = newText
			newState = []
			for p in combi:
				self.candidateHorizon[p] = "1"
				if not("_NIL" in self.candidates[p]):
					newState.append([newStateID, p])
			state_choices.append(newState)
			self.state_dic[newStateID] = combi
			self.stateText[newStateID] = self.stateText[currentTweetID]
			self.tweetid[newStateID] = currentTweetID
			combiIDcounter += 1
		return state_choices, mychoice	

	def getDenom(self, results, t):
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
					avg_denom += float(res[0]) * float(self.expert_weight_dic[(res[3], res[2], dependency, t)])
		return avg_denom

	def computeItSimple(self, results, t, label, mode, denom):
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
					* float(dor_impact) * float(self.expert_weight_dic[(res[3], res[2], dependency, t)]))			
		return avg

	def execute(self, chosen, h):
		candis = set()
		candis_1 = set()
		expert = chosen[0]
		sid = chosen[1]
		tweet = self.stateText[sid]
		pairs = self.datapoints[self.tweetid[sid]][3]
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
			if "<entity>" in self.textWithEntites[sid]:
				resultset = getFromFile("11",self.textWithEntites[sid])		
				if resultset == None:
					resultset = executeAGDISTIS('11', self.textWithEntites[sid])
					save2File("11", self.textWithEntites[sid], resultset)
		if expert == "12":
			if "<entity>" in self.textWithEntites[sid]:
				resultset = getFromFile("12",self.textWithEntites[sid])
				if resultset == None:
					resultset = executeAIDATagger('12', self.textWithEntites[sid])
					save2File("12", self.textWithEntites[sid], resultset)
		if expert == "13":
			if "<entity>" in self.textWithEntites[sid]:
				resultset = getFromFile("13",self.textWithEntites[sid])
				if resultset == None:
					resultset = executeSpotlightTagger('13', self.textWithEntites[sid],
					 self.state_dic[sid], self.candidates, self.beginCandidates, 0.3, 0)
					save2File("13", self.textWithEntites[sid], resultset)
		try:
			prior = self.executed[sid]
			prior.append(expert)
			self.executed[sid] = prior
		except:
			prior = [expert]
			self.executed[sid] = prior
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
					self.beginCandidates[candid0] = result[2]
					self.endCandidates[candid0] = result[3]
					self.candidates[candid0] = winner
					self.textid[candid0] = sol
					self.stateText[candid0] = tweet
					self.stateText[candid] = tweet
					self.candidates[candid] = ne1
					self.candidateHorizon[candid0] = "0.1"
					self.candidateHorizon[candid] = "1"
					self.horizon[sol] = '0'
					self.textid[candid] = sol
					self.beginCandidates[candid] = result[2]
					self.endCandidates[candid] = result[3]	
					if not(candid in tempcandidates):
							tempcandidates.append(candid)
							tempcandidatesWONIL.append(candid)					
					hypo = []
					hypo.append(sol)
					hypo.append(candid0)   
					hypo.append('0000' + result[0][0] + candid + sol)         
					hypo.append(candid)
					hypo.append(result[0][0])  
					self.hypos.append(hypo)
					currentArr = []
					try:
						currentArr = self.candidatesBy[candid]
					except KeyError:
						pass
					currentArr.append(expert)	
					self.candidatesBy[candid] = currentArr
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
								self.candidateBeliefsArr.append(cb)
								reward.append(1)
								self.rewards.append(reward)
								break
					if(positive == 0):
						reward.append(0.0)
						retrieved -= 1
						self.rewards.append(reward)
						cb = []
						cb.append(sol)
						cb.append(candid)
						cb.append(0.33)
						self.candidateBeliefsArr.append(cb)	
					reward.append(self.tweetid[sol])
					reward.append(len(pairs))
					try:		
						content = self.allrewards[sid]
						content.append(reward)
						self.allrewards[sid] = content
					except:
						content = [reward]
						self.allrewards[sid] = content
				else:
					candid2 = -1
					candid1 = -1
					res = ""
					res_enc = ""
					allCandidates = self.state_dic[state]
					ambig = -1
					rx = re.compile('\W+')
					for ca in allCandidates:
						c1 = self.candidates[ca].lower()
						c2 = result[1].lower()
						c1 = rx.sub(' ', c1)
						c2 = rx.sub(' ', c2)
						if c1 == c2:
							candid1 = ca
							ambig += 1
					if ambig == -1:
						for ca in allCandidates:
							c1 = self.candidates[ca].lower()
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
							print("->" +self.candidates[ca])
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
					h = hashlib.sha256(res + str(self.beginCandidates[candid1]) 
						+ self.endCandidates[candid1])
					h.hexdigest()
					candis.add((sol, candid1))
					candis_1.add(candid1)
					candid2 = int(h.hexdigest(),base=16)
					candid2 = str(candid2) + self.tweetid[state]	
					self.beginCandidates[str(candid2)] = self.beginCandidates[candid1]							 
					self.endCandidates[str(candid2)] = self.endCandidates[candid1]	
					self.candidates[str(candid2)] = res
					self.textid[str(candid2)] = sol
					self.stateText[str(candid2)] = self.stateText[sol]
					if candid1 == -1 or candid2 == -1:
						continue
					self.stateText[candid1] = self.textWithEntites[state].count("<entity>")	
					if not(str(candid2) in tempcandidates):
						tempcandidates.append(str(candid2))
						tempcandidatesWONIL.append(str(candid2))
					self.horizon[state] = '1'
					hypo = []
					self.candidateHorizon[candid1] = "1"
					self.candidateHorizon[str(candid2)] = "2"
					currentArr = []
					try:
						currentArr = self.candidatesBy[str(candid2)]
					except KeyError:
						pass
					currentArr.append(result[0][0])	
					self.candidatesBy[str(candid2)] = currentArr	
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
					self.rewards.append(reward)
					reward.append(self.tweetid[sol])
					reward.append(len(pairs))
					try:
						content = self.allrewards[state]
						content.append(reward)
						self.allrewards[state] = content
					except:
						content = [reward]
						self.allrewards[state] = content
		relevantCandidates = dict()			
		for hypo in self.rewards:
			if hypo[0] == sol and hypo[0] != hypo[1]:
				relevantCandidates[hypo[1]] = hypo[6]
		for rel in relevantCandidates:
			for algoID in self.executed[sid]:
				nil = 1
				for hypo in self.rewards:
					if hypo[1] == rel:
						if hypo[2] == algoID:
							nil = 0
				if nil == 1:
					candid9 = rel + "0000"	
					for candilies in self.candidates:
						if (self.candidates[candilies] == self.candidates[rel] + "_NIL" 
						and self.textid[candilies] == sol):
							candid9 = candilies
					if not(candid9 in tempcandidates):
						tempcandidates.append(candid9)
					self.beginCandidates[candid9] = self.beginCandidates[rel]
					self.endCandidates[candid9] = self.endCandidates[rel]
					self.candidates[candid9] = self.candidates[rel] + "_NIL"
					self.textid[candid9] = self.textid[rel]
					self.stateText[candid9] = tweet
					self.candidateHorizon[candid9] = self.candidateHorizon[rel]
					currentArr = []
					try:
						currentArr = self.candidatesBy[candid9]
					except KeyError:
						pass
					currentArr.append(algoID)	
					self.candidatesBy[candid9] = currentArr
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
						if (len(get_overlap(self.candidates[rel],pair[0])) >= (len(pair[0]) - 1)
						 and len(get_overlap(self.candidates[rel],pair[0]))) >= 1:
							receivedReward = 0.0
					reward.append(receivedReward)
					if True:
						print("NOT NOT NOT: (" +str(algoID) + ") " +str(self.candidates[candid9])
						 + " - " +str(receivedReward))
						self.rewards.append(reward)
						content = self.allrewards[sid]
						content.append(reward)
						self.allrewards[sid] = content
						self.candidateBeliefs[str(candid9)] = receivedReward
						self.hasReward['0000' + algoID + rel + sol] = receivedReward
						reward.append(self.tweetid[sol])
						reward.append(len(pairs))
		if h == "0":
			self.stateCandis[sol + "2222222222"] = candis_1
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
			self.rewards.append(reward2)
		return candis

	def setupBounds(self):
		self.bounds[('SameLingualType', "0")] = float(self.lower_bound) / 16.0 
		self.bounds[('CandidateWordEmbeddings', "0")] = float(self.lower_bound) / 35.0
		self.bounds[('CandidateTextMetaSimilarity', "0")] = float(self.lower_bound) / 35.0
		self.bounds[('StateTextLengthMetaSimilarity', "0")] = float(self.lower_bound) / 5.0 
		self.bounds[('StateCandidateWordEmbeddings', "0")] = float(self.lower_bound)  / 40.0
		self.bounds[('StateSameLingualType', "0")] = float(self.lower_bound) / 10.0
		self.bounds[('StateTextExtraChars', "0")] = float(self.lower_bound)

		self.bounds[('SameLingualType', "0.1")] = self.bounds[('SameLingualType', "0")]
		self.bounds[('CandidateWordEmbeddings', "0.1")] = self.bounds[('CandidateWordEmbeddings', "0")]
		self.bounds[('CandidateTextMetaSimilarity', "0.1")] = self.bounds[('CandidateTextMetaSimilarity', "0")]
		self.bounds[('StateTextLengthMetaSimilarity', "0.1")] = self.bounds[('StateTextLengthMetaSimilarity', "0")]
		self.bounds[('StateCandidateWordEmbeddings', "0.1")] = self.bounds[('StateCandidateWordEmbeddings', "0")]
		self.bounds[('StateSameLingualType', "0.1")] = self.bounds[('StateSameLingualType', "0")]
		self.bounds[('StateTextExtraChars', "0.1")] = self.bounds[('StateTextExtraChars', "0")]

		self.bounds[('SameLingualType', "1")] = self.bounds[('SameLingualType', "0")]
		self.bounds[('CandidateWordEmbeddings', "1")] = self.bounds[('CandidateWordEmbeddings', "0")]
		self.bounds[('CandidateTextMetaSimilarity', "1")] = self.bounds[('CandidateTextMetaSimilarity', "0")]
		self.bounds[('StateTextLengthMetaSimilarity', "1")] = self.bounds[('StateTextLengthMetaSimilarity', "0")]
		self.bounds[('StateCandidateWordEmbeddings', "1")] = self.bounds[('StateCandidateWordEmbeddings', "0")]
		self.bounds[('StateSameLingualType', "1")] = self.bounds[('StateSameLingualType', "0")]
		self.bounds[('StateTextExtraChars', "1")] = self.bounds[('StateTextExtraChars', "0")]

	def explore_online_hedge(self):
		self.setupBounds()
		mode = "train"
		self.datapoints = formatDataset(self.dataset_mode)
		print("LEN " +str(len(self.datapoints)))
		self.datapoints_set = random.sample(self.datapoints, self.limit) # in case of Spotlight: 59
		past_points = dict()
		H = ['0', '1']
		t = 1
		#####init dependencies #####################################################
		self.dependencies.append("LiveDOM")
		for expert in self.experts_dic["0"]:
			self.expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
			for b in self.behav["DOI"]:
				self.dependencies.append(("LiveDOI", b, expert))
		for expert in self.experts_dic["1"]:
			self.expert_weight_dic[(expert, "all", "LiveDOM", t)] = 1.0
			for b in self.behav["DOI"]:
				self.dependencies.append(("LiveDOI", b, expert))
		for expert in self.experts_dic["1"]:
			for b in self.behav["DOR"]:
				self.dependencies.append(("LiveDOR", b, expert))
		self.initWeights(self.experts_dic['0'], self.metasims_dic['0.1'], self.dependencies, str(t))
		self.initWeights(self.experts_dic['1'], self.metasims_dic['1'], self.dependencies, str(t))
		#####start learning #######################################################
		for point in self.datapoints_set:
			self.tweetid[point] = point
			text = self.datapoints[point][0]
			self.textid[point] = point
			labels = self.datapoints[point][3]
			self.candidateHorizon[point] = "0"
			self.candidates[point] = text
			self.stateText[point] = text
			weighted_samples = dict()
			weighted_samples['0'] = [[[point, point]]]
			for h in H:
				print("+-+-+-+-+- h=" +str(h) +" +-+-+-+-+-+-")
				for ws in weighted_samples[h]:
					for candi in ws:
						self.doMS(candi[1], h, self.metasims_dic[h])
				self.getNeigbs()
				results1 = dict()
				if h == "0":
					results1 = self.apriori_assess(self.experts_dic[h], self.experts_dic[str(int(h)+1)], 
						weighted_samples[h], self.metasims_dic[h], h, self.bounds)
				else:
					results1 = self.apriori_assess(self.experts_dic[h], [], weighted_samples[h], 
						self.metasims_dic[h], h, self.bounds)
				candis, chosen_experts = self.choose_and_execute(self.budget[h], results1, h, str(t), 
					self.metasims_dic[h])
				if h == "0":
					h = "0.1"
					for ca in candis:
						self.doMS(ca[1],h, self.metasims_dic[h])
				self.getNeigbs()
				results2 = dict()
				us = None
				pds = None
				results_dep = None
				Distr = None
				if h == "0" or h == "0.1":
					temph = "0"
					results2, us, pds, results_dep, Distr = self.dealWithWeights_pre(candis, 
						self.experts_dic[h], self.experts_dic[str(int(temph)+1)], self.metasims_dic[h],
						 str(t), self.bounds, h, 0.2, "online", self.dependencies, chosen_experts)
				else:
					results2, us, pds, results_dep, Distr = self.dealWithWeights_pre(candis,
					 self.experts_dic[h], [], self.metasims_dic[h], str(t), self.bounds, h, 0.2, "online",
					  self.dependencies, chosen_experts)
				newh = -1
				if float(h) == 0.1:
					newh = 1
				else:
					newh = int(h) + 1
				weighted_samples[str(newh)], winna = self.getNewSamples(self.metasims_dic[h], h, 1,
				 0.2, point, str(t), self.bounds, results2, mode)
				self.dealWithWeights_post(us, pds, t, self.metasims_dic[h], mode, self.dependencies, 
					self.experts_dic[h], winna, chosen_experts)
			past_points[point] = 1	
			if len(past_points) == self.limit:
				break
			if len(past_points) == self.lower_bound:
				mode = "test"	
			t += 1