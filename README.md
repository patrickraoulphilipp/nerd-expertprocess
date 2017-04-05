NERD Expert Process
===
Algorithms for expert processes for named entity recognition and disambiguation (NERD) based on 

	@inproceedings{philipp2017,
	 author    = {Patrick Philipp and Achim Rettinger},
	 title     = {Reinforcement Learning for Multi-Step Expert Advice},
	 booktitle = {AAMAS},
	 year      = {2017},
	 pages     = {to appear}

Instricutions:
-------------
 * Download wvlib (https://github.com/spyysalo/wvlib) and trained embeddings (e.g. from https://code.google.com/archive/p/word2vec/)
 * Create folder "stash" in main dir
 * Call test.py with parameters embedding file, no. training samples, no. test samples, budget NER pre, budget NER post, budget NED pre, Microposts || Spotlight, e.g.: 
 	python test.py GoogleNews-vectors-negative300.bin 200 200 230 2 4 Microposts