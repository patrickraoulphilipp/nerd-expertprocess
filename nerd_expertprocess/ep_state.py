from sklearn.neighbors import NearestNeighbors
from datasketch import MinHashLSH

clip_model, clip_device = None, None

global_neighbors = dict()
reward_collector = dict()
inv_reward_collector = dict()
candidate_vectors = dict()
state_vectors = dict()
candidates = dict()
candidate_beliefs = dict()
text_id = dict() 
state_text = dict()
begin_candidates = dict()
end_candidates = dict()
text = dict()
tweetid = dict()
text_with_entites = dict()
hypos = []
rewards = []
executed = dict()
datapoints = dict()
candidate_horizon = dict()
horizon = dict()
candidates_by = dict()
all_rewards = dict()

state_dic = dict()
candidate_rewards = dict()
expert_weight_dic = dict()

Y_Label_count = 0
Y_Labels = []
Y_Preds = []
Y_Label = []
Y_Label_global = []
Y_Pred = []
Y_Pred_global = []
Y_Pred_count = 0
Y_Pred_1 = []
Y_Label_1 = []
Y_Label_count_1 = 0
Y_Label_1 = []
Y_Pred_count_1 = 0
Y_Pred_Singles = []
Y_Label_Singles = []
Y_Pred_Singles_1 = []
Y_Label_Singles_1 = []
Y_theo = []
Y_Label_text = []
Y_Label_text_1 = []
chosen_exps = []

failed_experts = 0
failed_points = 0

minhash_sim = dict()
minhash_sim1 = dict()
posMapper = dict()
state_emb_sim = dict()
embedded_sim = dict()
neighbors = dict()
ling_vector = dict()
X_length = []
Y_length = []
X_extra = []
Y_extra = []
X_ling = []
Y_ling = []
M_rev = dict()
M = dict()
M_rev_1 = dict()
M1 = dict()

model = None
device = None

nbrs_length = NearestNeighbors(radius=30)
nbrs_extra = NearestNeighbors(radius=2)
nbrs_ling = NearestNeighbors(radius=0)
mc = 0
mc1 = 1
lsh_0 = MinHashLSH(threshold=0.2, num_perm=128)
lsh_1 = MinHashLSH(threshold=0.2, num_perm=128)
