import os
import random
import shutil
import time
import pickle
import numpy as np
from predictor import LSTM_predictor
from control_policy import ChlorinationControlPolicyNeat
from scenarios import load_scenario
from env import WaterChlorinationEnv
from glob import glob
import os
import neat
import multiprocessing as mp
import tqdm
import gzip
import pandas as pd

class ChlorinationControlPolicyNeatSurrogate(ChlorinationControlPolicyNeat):
    def __init__(self, env, config_path, predictor, context_data_folder,
                 lookback_days=5, scenario_ids=None, save_path="neat", nsga2_use=True):
        super().__init__(env, config_path, scenario_ids or [], save_path, nsga2_use)

        self.predictor = predictor
        self.lookback_days = lookback_days
        self.context_dataset = self.load_contexts(context_data_folder)

        if self.context_dataset:
            self.obs_dim = self.context_dataset[0].shape[1]
            self.action_dim = self.predictor.X_action.shape[2]

    def load_contexts(self, folder_path):
        context_seqs = []
        for file in glob(os.path.join(folder_path, "*.csv")):
            df = pd.read_csv(file, header=None, names=["scenario_id", "obs", "action", "reward"])
            df['obs'] = df['obs'].apply(lambda x: np.array([float(i) for i in str(x).split(';')]))

            obs_seq = np.stack(df['obs'].to_numpy())
            if obs_seq.shape[0] < self.lookback_days:
                continue

            for i in range(self.lookback_days, obs_seq.shape[0]):
                context_seqs.append(obs_seq[i - self.lookback_days:i])
        return context_seqs

    def eval_genome_surrogate(self, genome_tuple, config):
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
        # from predictor import LSTM_predictor

        # genome_id, genome = genome_tuple
        # net = neat.nn.FeedForwardNetwork.create(genome, config)

        # predictor = LSTM_predictor(
        #     data_path=None,  # No need to train again
        #     window_size=10,
        #     lookback_days=5,
        #     lstm_size=64
        # )
        # predictor.load(pred_path, scaler_path)

        genome_id, genome = genome_tuple
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        predicted_rewards = []
        sampled_data = random.sample(self.context_dataset, min(50, len(self.context_dataset)))
        for context_seq in tqdm.tqdm(sampled_data, total=len(sampled_data), desc=f'gid={genome_id}'):
            action_history = []

            for t in range(self.lookback_days):
                obs = context_seq[t]
                act = net.activate(obs.tolist())
                action_history.append(act)

            context_arr = np.array(context_seq)  # (lookback_days, obs_dim)
            action_arr = np.array(action_history)  # (lookback_days, action_dim)

            try:
                pred_reward = self.predictor.predict(context_arr, action_arr)
            except Exception as e:
                print(f"surrogate prediction failed for genome {genome_id}: {e}")
                pred_reward = -9999

            predicted_rewards.append(pred_reward)

        avg_reward = np.mean(predicted_rewards)
        print(genome_id, avg_reward)
        fitness = avg_reward

        return genome_id, fitness

    def eval_genomes(self, genomes, config):
        params = [[genome, config] for genome in genomes]
        cpus = np.min([mp.cpu_count(), len(params), 4])
        print(f'cpus: {cpus}, num ind: {len(params)}')
        genomes_dict = {
            genome[0]: genome[1]
            for genome in genomes
        }

        print(f"Evaluating {len(params)} genomes using surrogate model...")

        # with mp.Pool(cpus) as p:
        #     r_genomes = p.starmap(self.eval_genome_surrogate, params)
        #     p.close()
        #     p.join()

        for g, c in params:
            gid, fitness = self.eval_genome_surrogate(g, c)
            genomes_dict[gid].fitness = fitness

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genomes_dict[gid]
                with open(f"{self.save_path}/best_genome.pkl", "wb") as f:
                    pickle.dump(self.best_genome, f)

def run_genomeenv(it, genome, config, scenario_ids):
    from evaluation import evaluate

    genome_id, genome = genome
    print(it, genome_id, scenario_ids)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitness_s = np.zeros((len(scenario_ids),5))
    for i, sid in enumerate(scenario_ids):
        with WaterChlorinationEnv(**load_scenario(scenario_id=sid)) as env:    
            stime = time.time()
            obs, info = env.reset()
            print(f'gid={genome_id}, loaded {sid}; {time.time()-stime:.2f} s', flush=True)

            policy = ChlorinationControlPolicyNeat._from_net(env, net)

            try:
                score = evaluate(policy, env)
            except Exception as e:
                print('exception', e, flush=True)
                # import traceback
                # traceback.print_exc()
            obj = -np.array([
                score['cost_control'],
                score['bound_violations'],
                score['bound_violations_fairness'],
                score['injection_smoothness_score'],
                score['infection_risk'][0]
            ])
            fitness_s[i] += obj
            print(f'gid={genome_id}, s_id={sid} : fitness={fitness_s[i]}; time={time.time()-stime}', flush=True)

            # ['scenario_id', 'obs', 'action', 'reward'] 
            score['report'] = [[sid]+r for r in score['report']]
            with open(f'data/it_{it+1}/data_{genome_id}.csv', 'a') as f:
                res = "\n".join(
                    [",".join(list(map(str, x))) for x in score['report']]
                )
                f.write(res+'\n\n')
 
    avg_fitness = np.sum(fitness_s, axis=0)
    return (genome_id, avg_fitness) 

def run_best_prescriptors_in_env(best_genomes, config, iteration, n_to_run=3):
    # sampled = random.sample(SCENARIO_POOL, k=10)
    sampled = list(range(1,10))
    # sampled = [1,2]
    os.makedirs(os.path.join(DATA_DIR, f'it_{iteration+1}'), exist_ok=True)
    print(f"Running best prescriptors on real env in scenarios: {sampled}")

    params = [[iteration, [i, genome], config] for i, genome in enumerate(best_genomes)]
    cpus = np.min([mp.cpu_count(), len(best_genomes)])
    print(f'cpus: {cpus}, num ind: {len(params)}')

    # genomes_d = {
    #     genome[0]: genome[1]
    #     for genome in best_genomes
    # }

    # for scenario_id in sampled:
    params = [x + [sampled] for x in params]
    with mp.Pool(cpus) as p:
        r_genomes = p.starmap(run_genomeenv, params)
        p.close()
        p.join()

        # for gid, fitness in r_genomes:
        #     genomes_d[gid].fitness = fitness

        #     if fitness > self.best_fitness:
        #         self.best_fitness = fitness
        #         self.best_genome = genomes_d[gid]
        #         with open(f"{self.save_path}/best_genome.pkl", "wb") as f:
        #             pickle.dump(self.best_genome, f)

def load_top_genomes(save_path, config, top_n=3):
    genome_files = glob(os.path.join(save_path, 'neat-checkpoint-*'))
    genome_files.sort(key=os.path.getmtime, reverse=True)
    file = genome_files[0]
    top_genomes = []

    # for file in genome_files:
    print(f"Loading checkpoint: {file}")
    with gzip.open(file, 'rb') as f:
        generation, loaded_config, population, species_set, rndstate = pickle.load(f)
        print(f"Checkpoint generation: {generation}, population size: {len(population)}")

        all_genomes = list(population.values())
        all_genomes = [g for g in all_genomes if g.fitness is not None]
        print(f"{len(all_genomes)} genomes have fitness")

        sorted_genomes = sorted(all_genomes, key=lambda g: g.fitness, reverse=True)
        top_genomes.extend(sorted_genomes[:top_n])

    print(f"Returning top {len(top_genomes[:top_n])} genomes")
    return top_genomes[:top_n]


def run_pipeline_loop():
    print("========== Starting pipeline ==========")
    iteration = 0
    dummy_env = WaterChlorinationEnv(**load_scenario(scenario_id=1))
    surrogate_policy = None

    while True:
        print(f"\n=== Iteration {iteration} ===")

        # Load and train predictor
        predictor = LSTM_predictor(
            data_path=DATA_DIR+f'/it_{iteration}',
            window_size=10,
            lookback_days=NB_LOOKBACK_DAYS,
            lstm_size=64,
            save_path='./'
        )

        pred_exists = os.path.exists(PREDICTOR_WEIGHTS) and os.path.exists(REWARD_SCALER)
        if pred_exists:
            predictor.load_model_and_scaler(PREDICTOR_WEIGHTS, REWARD_SCALER)

        if not (iteration == 0 and pred_exists):
            predictor.train(
                epochs=40,
                batch_size=128,
                verbose=2
            )

        if iteration == 0:
            surrogate_policy = ChlorinationControlPolicyNeatSurrogate(
                env=dummy_env, # not used
                config_path=CONFIG_PATH,
                predictor=predictor,
                context_data_folder=DATA_DIR+f'/it_{iteration}',
                lookback_days=NB_LOOKBACK_DAYS,
                scenario_ids=[], # not used
                save_path=SAVE_PATH,
                nsga2_use=False
            )
        else:
            genome_files = glob(os.path.join(SAVE_PATH, 'neat-checkpoint-*'))
            genome_files.sort(key=os.path.getmtime, reverse=True)
            print(f'Loading neat pop. from {genome_files[0]}')
            surrogate_policy.train(
                n_generations=1,
                p=neat.checkpoint.Checkpointer.restore_checkpoint(genome_files[0])
            )

        surrogate_policy.train(n_generations=50)

        top_genomes = load_top_genomes(SAVE_PATH, surrogate_policy.config, top_n=TOP_N_TO_EVALUATE)
        print(top_genomes)
        
        run_best_prescriptors_in_env(
            top_genomes, 
            surrogate_policy.config, 
            iteration=iteration, 
            n_to_run=3
        )
        
        time.sleep(10)
        iteration += 1

DATA_DIR = './data'
CONFIG_PATH = './neat-config.init'
SAVE_PATH = './neat'
PREDICTOR_WEIGHTS = 'model.weights.h5'
REWARD_SCALER = 'reward_scaler.pkl'

NB_LOOKBACK_DAYS = 10
TOP_N_TO_EVALUATE = 5
SCENARIO_POOL = list(range(1, 10))  # 1 through 9


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_pipeline_loop()
