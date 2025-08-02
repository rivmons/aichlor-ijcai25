"""
This module contains the base class of control policies and a random baseline policy.
"""
from abc import abstractmethod
import numpy as np
import neat
from env import WaterChlorinationEnv
from scenarios import load_scenario
import pickle
import multiprocessing as mp
import os
import shutil
import time
import pickle
from nsga2 import NSGA2Reproduction, NSGA2Fitness


class ChlorinationControlPolicy():
    """
    Base class of control policies.
    """
    def __init__(self, env: WaterChlorinationEnv):
        if not isinstance(env, WaterChlorinationEnv):
            raise TypeError("'env' must be an instance of 'WaterChlorinationEnv' "+
                            f"but not of '{type(env)}'")

        self._gym_action_space = env.action_space

    @abstractmethod
    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        """
        Computes and returns an action based on a given observation (i.e. sensor readings).

        Parameters
        ----------
        observations : `numpy.ndarray`
            Observation (i.e. sensor readings)

        Returns
        -------
        `numpy.ndarray`
            Actions (i.e. chlorine injection at each injection pump)
        """
        raise NotImplementedError()

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return self.compute_action(observations)


class ChlorinationControlPolicyRandom(ChlorinationControlPolicy):
    """
    Random control policy -- picks a random control action in every time step.
    """
    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        return self._gym_action_space.sample()

class ChlorinationControlPolicyNeat(ChlorinationControlPolicy):
    def __init__(self, env, config_path, scenario_ids=range(10), save_path="neat", nsga2_use=False):
        super().__init__(env)
        self.config_path = config_path
        self.config = None

        self.nsga2 = nsga2_use

        self.net = None  # Best neural net after training
        self.scenario_ids = list(scenario_ids)
        self.save_path = save_path
        self.best_fitness = -np.inf

        self.envs = []

    def compute_action(self, observations: np.ndarray) -> np.ndarray:
        if self.net is None:
            return self._gym_action_space.sample()
        output = self.net.activate(observations.tolist())
        return np.array(output) 
    
    def eval_genome(self, genome, config):
        from evaluation import evaluate
        genome_id, genome = genome
        genome.fitness = NSGA2Fitness(np.zeros(5,))
        # reports = []
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness_s = np.zeros((len(self.scenario_ids),5))
        for scenario_id in self.scenario_ids:
            with WaterChlorinationEnv(**load_scenario(scenario_id=scenario_id)) as env:    
                stime = time.time()
                obs, info = env.reset()
                print(f'gid={genome_id}, loaded {scenario_id}; {time.time()-stime} s', flush=True)
                self.net = net
                try:
                    score = evaluate(self, env)
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
                fitness_s[scenario_id-self.scenario_ids[0]] += obj
                # print(obj, fitness_s[scenario_id-self.scenario_ids[0]])
                # print(score, flush=True)
                # print(f'gid={genome_id}, s_id={scenario_id} : fitness=[{", ".join(f"{x:.6f}" for x in obj)}]; time={time.time()-stime}', flush=True)

                # ist = 0
                # while True:
                #     # Apply policy
                #     act = net.activate(obs)
                #     obs, reward, terminated, _, _ = env.step(act)
                #     reward = -np.mean(act)
                #     # print(ist, reward, terminated, flush=True)
                #     ist += 1
                #     if terminated is True:
                #         break

                    # fitness_s[scenario_id-self.scenario_ids[0]] += reward
                
                print(f'gid={genome_id}, s_id={scenario_id} : fitness={fitness_s[scenario_id-self.scenario_ids[0]]}; time={time.time()-stime}', flush=True)
                score['report'] = [[scenario_id]+r for r in score['report']]
                print(score['report'])
                with open(f'data/data_{genome_id}.csv', 'a') as f:
                    res = "\n".join(
                        [",".join(list(map(str, x))) for x in score['report']]
                    )
                    f.write(res+'\n\n')
        # print(fitness_s.shape, np.sum(fitness_s, axis=0), np.sum(fitness_s, axis=1))
        # with open(f'data/data_{genome_id}.csv', 'a') as f:
        #     res = "\n\n".join(
        #         "\n".join(",".join(map(str, inner)) for inner in outer)
        #         for outer in reports
        #     )
        #     f.write(res+'\n\n')
        avg_fitness = np.sum(fitness_s, axis=0) / len(self.scenario_ids)
        genome.fitness.add(avg_fitness)
        return (genome_id, genome.fitness) 
    
    # Function that evaluates the fitness of each prescriptor model
    def eval_genomes(self, genomes, config):
        params = [[genome, config] for genome in genomes]
        cpus = np.min([mp.cpu_count(), len(params), 25])
        print(f'cpus: {cpus}, num ind: {len(params)}')
        genomes_d = {
            genome[0]: genome[1]
            for genome in genomes
        }
        
        with mp.Pool(cpus) as p:
            r_genomes = p.starmap(self.eval_genome, params)
            p.close()
            p.join()

        # r_genomes = []
        # for g in params:
        #     r_genomes.append(self.eval_genome(*g))

        for gid, fitness in r_genomes:
            genomes_d[gid].fitness = fitness

            # Save best genome found so far
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_genome = genomes_d[gid]
                with open(f"{self.save_path}/best_genome.pkl", "wb") as f:
                    pickle.dump(self.best_genome, f)

    def train(self, n_generations=50, p=None):
        # Create the population, which is the top-level object for a NEAT run.
        # p = None
        if not p:
            if self.nsga2:
                self.config = neat.Config(
                    neat.DefaultGenome, 
                    NSGA2Reproduction,
                    neat.DefaultSpeciesSet, 
                    neat.DefaultStagnation,
                    self.config_path)
                p = neat.Population(self.config)
            else:
                self.config = neat.Config(
                    neat.DefaultGenome,
                    neat.DefaultReproduction,
                    neat.DefaultSpeciesSet,
                    neat.DefaultStagnation,
                    self.config_path
                )
                p = neat.Population(self.config)

        p.add_reporter(neat.StdOutReporter(show_species_detail=True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(generation_interval=1,
                                        time_interval_seconds=600,
                                        filename_prefix=f'{self.save_path}/neat-checkpoint-'))
        
        shutil.rmtree(self.save_path, ignore_errors=True)
        os.makedirs(self.save_path)

        winner = p.run(self.eval_genomes, n_generations)
        # print("Training complete. Best genome:")
        # print(winner)

        self.net = neat.nn.FeedForwardNetwork.create(winner, self.config)

    @classmethod
    def _from_net(cls, env, net):
        inst = cls.__new__(cls)
        ChlorinationControlPolicy.__init__(inst, env)
        inst.net = net
        return inst

    def load(self, genome_path):
        """Load a saved genome from disk and create the network."""
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(genome, self.config)
