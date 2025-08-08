import numpy as np
import neat
import glob
import os
import pickle
import gzip
import time
import multiprocessing as mp

from control_policy import ChlorinationControlPolicyNeat
from scenarios import load_scenario
from env import WaterChlorinationEnv

def eval_g(genome, config, scenario_ids):
        from evaluation import evaluate

        genome_id, genome = genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        fitness_s = np.zeros((len(scenario_ids),5))
        for i, sid in enumerate(scenario_ids):
            with WaterChlorinationEnv(**load_scenario(scenario_id=sid)) as env:    
                stime = time.time()
                obs, info = env.reset()
                print(obs.shape)
                print(f'gid={genome_id}, loaded {sid}; {time.time()-stime:.2f} s', flush=True)

                policy = ChlorinationControlPolicyNeat._from_net(env, net)

                try:
                    score = evaluate(policy, env)
                except Exception as e:
                    print('exception', e, flush=True)

                obj = -np.array([
                    score['cost_control'],
                    score['bound_violations'],
                    score['bound_violations_fairness'],
                    score['injection_smoothness_score'],
                    score['infection_risk'][0]
                ])
                fitness_s[i] += obj
                print(f'gid={genome_id}, s_id={sid} : fitness={fitness_s[i]}; time={time.time()-stime}', flush=True)

                with open(f'pareto1.csv', 'a') as f:
                    f.write(f'{genome_id},{",".join(list(map(str, obj.tolist())))}\n')
    
        avg_fitness = np.sum(fitness_s, axis=0)
        return (genome_id, avg_fitness) 


def eval_best(best_genomes):
    sampled = [3]
    print(best_genomes)
    params = [[[id, genome], config, sampled] for id, (genome, config) in best_genomes.items()]

    if not os.path.exists('pareto1.csv'):
        with open(f'pareto1.csv', 'a') as f:
            f.write(f'gid,cost_control,bound_violations,fairness,smoothness,infection_risk\n')

    cpus = np.min([mp.cpu_count(), len(best_genomes)])
    print(f'cpus: {cpus}, num ind: {len(params)}')

    with mp.Pool(cpus) as p:
        r_genomes = p.starmap(eval_g, params)
        p.close()
        p.join()

def load_checkpoint_pops(save_path, n_gen=3):
    genome_files = glob.glob(os.path.join(save_path, 'neat-checkpoint-*'))
    genome_files.sort(key=os.path.getmtime, reverse=True)
    top_genomes = {}

    for file in genome_files[:n_gen]:
        print(f"Loading checkpoint: {file}")
        with gzip.open(file, 'rb') as f:
            generation, config, population, species_set, rndstate = pickle.load(f)
            print(f"Checkpoint generation: {generation}, population size: {len(population)}")
            top_genomes.update({
                k: [v, config]
                for k, v in population.items()
            })
    
    return top_genomes

def pareto_analysis(csv_path, normalize=True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pandas.plotting import parallel_coordinates
    from itertools import combinations

    def prettify(name):
        return name.replace('_', ' ').title()

    metrics = ['cost_control','bound_violations','smoothness','infection_risk']

    normalize = True
    output_dir = 'pareto_2d'

    df = pd.read_csv(csv_path)
    data = df[metrics].copy()

    if normalize:
        data = (data - data.min()) / (data.max() - data.min())

    # === FIXED PARETO 2D FUNCTION ===
    def pareto_2d(xy):
        is_efficient = np.ones(xy.shape[0], dtype=bool)
        for i, point in enumerate(xy):
            if is_efficient[i]:
                others = np.delete(xy, i, axis=0)
                if np.any(np.all(others >= point, axis=1) & np.any(others > point, axis=1)):
                    is_efficient[i] = False
        return is_efficient

    os.makedirs(output_dir, exist_ok=True)

    pairs = list(combinations(metrics, 2))

    for x, y in pairs:
        xy = data[[x, y]].values
        pareto_mask = pareto_2d(xy)
        print(pareto_mask)
        print(df['gid'])

        #993

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(
            xy[~pareto_mask, 0], xy[~pareto_mask, 1],
            color='steelblue', alpha=0.4, label='Others'
        )
        ax.scatter(
            xy[pareto_mask, 0], xy[pareto_mask, 1],
            color='crimson', s=40, label='Pareto Front'
        )

        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.set_xlabel(prettify(x), fontsize=12)
        ax.set_ylabel(prettify(y), fontsize=12)
        ax.set_title(f'{prettify(x)} vs {prettify(y)}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        margin = 0.05  # 5% padding

        x_vals = xy[:, 0]
        y_vals = xy[:, 1]

        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        ax.set_xlim(max(0, x_max + margin), min(1, x_min - margin))
        ax.set_ylim(max(0, y_max + margin), min(1, y_min - margin))

        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'{x}_vs_{y}_pareto.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()

        print(f'Saved: {fig_path}')

if __name__ == "__main__":
    dg = load_checkpoint_pops('./direct-neat', n_gen=3)
    # with open('./submission/best_genome.pkl', 'wb') as f:
    #     pickle.dump(dg[1000], f)
    eval_best({993: dg[993]})

    # pareto_analysis('pareto1.csv', True)