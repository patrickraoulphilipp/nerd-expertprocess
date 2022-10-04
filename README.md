Expert Processes for Named Entity Recognition and Disambiguation
===
Implementation of the online hedge algorithm for NERD expert processes of *Reinforcement Learning for Multi-Step Expert Advice* using so-called meta-dependencies as features, which are neighorhood-based assessment of available expert services. Note that the expert services for NER and NED (e.g. AIDA, DBpedia Spotlight) are not accessible anymore, but I provide a "stash" of >300K service evaluations for the integrated datasets. I also added exemplary textual features based on OpenAI's ClIP model, which were not included in the original implementation.

Instructions
-------------
1.  Clone this repo.

```
git clone https://github.com/patrickraoulphilipp/nerd-expertprocess
cd nerd-expertprocess
```

2. (optional) Create a virtualenv. The implementation has been tested for Python 3.9.

```
virtualenv venv
source venv/bin/activate
```

3. Install all dependencies. You need CLIP, which will be automatically installed from the respective git repo.

```
pip install -r requirements.txt .
```

4. Download all nltk dependencies, which you can do via the python script in the scripts folder.

```
python scripts/install_nltk.py
```

5. (Optional but recommended) Download the zipped expert service stash from the following link.

```
Direct link: https://drive.google.com/file/d/1T94xTkOrm3gyJEB2FvK4PJnT8XqWykqC/view?usp=sharing
```

6. Set parameter **STASH_PATH** in nerd_expertprocess/ep_config.py, which should either point to an empty folder to gather the expert service results or to the downloaded & unzipped stash folder. 

```
STASH_PATH = '/PATH/TO/FOLDER/'
...
```

7. Run main.py to start the search process.

```
python main.py
```


Cite as
-------------
	@inproceedings{philipp2017,
	 author    = {Patrick Philipp and Achim Rettinger},
	 title     = {Reinforcement Learning for Multi-Step Expert Advice},
	 booktitle = {AAMAS},
	 year      = {2017},
	 pages     = {962--971}