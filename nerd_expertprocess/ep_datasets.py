import numpy as np
import io
from rdflib.graph import Graph

from nerd_expertprocess import ep_state
from nerd_expertprocess.ep_config import *

def format_dataset(data):
    datapoints = dict()
    if data == 'Microposts':
        with io.open(MICROPOST_PATH, 'r', encoding='utf-8') as f:
            total_ents = 0
            for l in f:
                # l = l.encode("ascii", "ignore")
                temp = l.strip().split("\t")
                datapoints[str(temp[0])] = "test"
                datapoints[str(temp[0])] = [temp[1], 'tweet', 'Microposts_2014',[]]
                a = np.array(temp)
                a = np.delete(a, [0,1])
                helper = []
                for item in a:
                    helper.append(item)
                    if len(helper) == 2:
                        pred0 = helper[0]
                        pred1 = helper[1]
                        aggr = [pred0, -1, -1, pred1]
                        total_ents += 1		
                        datapoints[str(temp[0])][3].append(aggr)
                        helper = []
            print("Micro has #tweets=" +str(len(datapoints)) + ", and #ents="+str(total_ents))
    if data == 'Spotlight':
        g = Graph()
        g.parse(SPOTLIGHT_PATH, format='n3')
        qres = g.query(
            """Prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#>
        Prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#>        
        SELECT ?x ?state_text  ?dne ?candidate ?start ?end ?candidatetext
        WHERE {
            ?x ?y nif:Sentence;
            nif:anchorOf ?state_text.
                ?candidate  nif:sentence       ?x;
                itsrdf:taIdentRef  ?dne;
                nif:beginIndex     ?start;
                nif:endIndex       ?end;
                nif:anchorOf       ?candidatetext.
               }""")
        for row in qres:			
            uri = row["x"].encode("ascii", "ignore")
            ep_state.state_text = row["state_text"].encode("ascii", "ignore")
            dne = row["dne"].encode("ascii", "ignore")
            start = row["start"].encode("ascii", "ignore")
            end = row["end"].encode("ascii", "ignore")
            candidatetext = row["candidatetext"].encode("ascii", "ignore")
            aggr = [candidatetext, start, end, dne]
            stuff = []
            try:
                stuff = datapoints[uri]
                current = stuff[3]
                current.append(aggr)
                stuff[3] = current
            except KeyError:
                pass
            if len(stuff) == 0:
                datapoints[uri] = [ep_state.state_text, 'NY Times', 'Spotlight', [aggr]]
    return datapoints