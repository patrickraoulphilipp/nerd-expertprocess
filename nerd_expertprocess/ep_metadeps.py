from nerd_expertprocess import ep_state

def calculate_relatedness_md(
        protagonist, 
        sid, 
        cid, 
        metasims, 
        h, 
        threshold, 
        bounds
):
    metasimarray = []
    for metasim in metasims:
        computeM = 0
        counter = 0
        c2 = 0
        total2 = 0.0
        fullcounter = 0
        neighs = []
        try:
            neighs = ep_state.global_neighbors[(cid,metasim,str(h))]
        except:
            pass
        for neigh in neighs:
            rews2 = []
            try:
                rews2 = ep_state.reward_collector[(protagonist,neigh[0])]
            except KeyError:
                continue
            for rew2 in rews2:
                if (rew2[2] == protagonist and sid != rew2[0] and cid != rew2[1] 
                and ep_state.tweetid != rew2[7]):
                    similar = 0
                    similar = neigh[1]
                    if float(similar) >= float(threshold):
                        computeM += float(similar)
                        weight = float(similar)
                        c2 += 1
                        total2 += (float(rew2[6]) * float(weight))
                        counter += weight
                        fullcounter += 1
        if not(counter == 0):
            computeM = float(computeM) / float(bounds[metasim, h])
            if computeM > 1.0:
                computeM = 1.0
            res2 = ( float(total2)/float(counter) )
            metasimarray.append([res2, computeM, metasim, protagonist])
        else:
            metasimarray.append([0, 0, metasim, protagonist])
    return metasimarray

def calculate_pairwise_md(
        protagonist, 
        competitor, 
        sid, 
        cid, 
        metasims, 
        h, 
        threshold, 
        bounds
):	
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
            ep_state.global_neighbors[(cid,metasim,str(h))]
        except:
            ep_state.global_neighbors[(cid,metasim,str(h))] = []
        for neigh in ep_state.global_neighbors[(cid,metasim,str(h))]:	
            rewhelpers = []
            try:
                rewhelpers = ep_state.reward_collector[(protagonist,neigh[0])]
            except KeyError:
                continue
            for rewhelper in rewhelpers: 
                if rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1
                ] and ep_state.tweetid != rewhelper[7]:
                    similar = float(neigh[1])	
                    if similar >= threshold:
                        rews2 = []
                        try:
                            rews2 = ep_state.reward_collector[(competitor,neigh[0])]
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
                                else:
                                    agree = 1
                                carr.append([float(similar) * agree, similar])
                                if rewhelper[4] == rew2[4]:
                                    confidenceAdjustedPerformanceRate.append([float(similar) 
                                        * float(performanceConf), similar])
                                else:
                                    performanceRate.append([float(similar) * float(rewhelper[6]), similar])
                                    agreementRate.append([float(similar) * float(rew2[6]), similar])
                                if not(float(rew2[6]) > 0.6 and float(rewhelper[4]) > 0.6):
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
        predictors.extend([[agreeresult, density, metasim, protagonist, "Agree", competitor], [confagreeresult,
        density, metasim, protagonist, "ConfAgree", competitor], [confperfresult, density, metasim, protagonist,
        "ConfPerf", competitor], [perfresult, density, metasim, protagonist, "Perf", competitor], [indresult,
        density, metasim, protagonist, "IndError", competitor]])
    return predictors

def calculate_robustess_md(
        protagonist, 
        competitor, 
        sid, cid, 
        metasims, 
        h, 
        threshold, 
        bounds
):
    predictors = []
    for metasim in metasims:
        performanceRate = []
        confidenceAdjustedPerformanceRate = []
        independentErrorRate = []
        computeM = 0
        ccounter = 0
        for neigh in ep_state.global_neighbors[(cid,metasim,str(h))]:
            rewhelpers = []
            try:
                rewhelpers = ep_state.reward_collector[(protagonist,neigh[0])]
            except KeyError:
                continue
            for rewhelper in rewhelpers:
                if (rewhelper[2] == protagonist and sid != rewhelper[0] and cid != rewhelper[1] 
                and ep_state.tweetid != rewhelper[7] and rewhelper[0] != rewhelper[1]):
                    similar = float(neigh[1])
                    if similar >= threshold:
                        computeM += float(similar)
                        ccounter += 1
                        rews2 = []	
                        try:
                            rews2 = ep_state.reward_collector[(competitor,rewhelper[4])]
                        except KeyError:
                            try:
                                if "_NIL" in ep_state.candidates[rewhelper[4]] and rewhelper[6] < 0.5:
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
        predictors.extend([ [indresult, density, metasim, protagonist, "IndError", competitor],
         [perfresult, density, metasim, protagonist, "Perf", competitor], 
         [confperfresult, density, metasim, protagonist, "ConfPerf", competitor]])
    return predictors