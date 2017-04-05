from ep_test import *
import sys

def main(argv=None):
	if argv is None:
		argv = sys.argv
	_lower_bound = int(argv[2])
	_limit = int(argv[3])
	_budget = dict()
	_budget["0"] = int(argv[4])
	_budget["0.1"] = int(argv[5])
	_budget["1"] = int(argv[6])
	_dataset_mode = argv[7]
	_options = argv[1:2]
	ep = EP(lower_bound=_lower_bound, limit=_limit, budget=_budget, dataset_mode=_dataset_mode, options=_options)
	ep.explore_online_hedge()

if __name__=='__main__':
    main(sys.argv)