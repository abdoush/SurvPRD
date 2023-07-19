from model import CPHExperimentSim, CPHExperimentCMPASS
from evolution import CPHMaxCindexGenetic
from helper import Results



max_iters = 1000

# select dataset

# Sim dataset
exp_dir = '25'
correct_labels_p=0.25
censoring_p_orig=0.50
n_samples=2000
dims_hazard_ratios=[1, 2, 3, 4]
baseline_hazard=0.1
random_state_seed=20
test_p = 0.5
exp = CPHExperimentSim(n_samples=n_samples,
                    censoring_p_orig=censoring_p_orig,
                    correctly_labeled_p=correct_labels_p,
                    test_p=test_p,
                    random_state_seed=random_state_seed)


# SMPASS dataset
# exp_dir = 'CMPASS'
# correct_labels_p=0.50
# censoring_p_orig=0.50
# test_p = 0.5
# exp = CPHExperimentCMPASS(censoring_p_orig=censoring_p_orig, correctly_labeled_p=correct_labels_p, test_p=test_p)

g = CPHMaxCindexGenetic(exp=exp, correct_labels_p=correct_labels_p, ind_mutation_p=0.5, cross_over_p=1, bit_mutation_p=0.1,
                 seed=0, find_in='EM', log_dir=f'results/{exp_dir}')

g.logger.info(f'Correct CI Train vs Test: {g.exp.correct_score}')
g.logger.info(f'Correct CI Test vs Train: {g.exp.correct_score_test}')

g.search_EM(population_size=100, max_generations=10000, max_nochange=30, max_iters=max_iters)

Results.save(g=g, exp_dir=exp_dir, correct_labels_p=correct_labels_p, censoring_p_orig=censoring_p_orig)