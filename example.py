"""
Example of how to use the starter code.
"""
from env import WaterChlorinationEnv
from scenarios import load_scenario
from control_policy import ChlorinationControlPolicyRandom, ChlorinationControlPolicyNeat
from evaluation import evaluate
import pickle
import multiprocessing as mp
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    with WaterChlorinationEnv(**load_scenario(scenario_id=0)) as env:
        # Create new random policy
        # TODO: Develop a "smarter" policy/controller
        # my_policy = ChlorinationControlPolicyRandom(env)

        # # Evaluate policy
        # r = evaluate(my_policy, env)
        # print(r)

        policy = ChlorinationControlPolicyNeat(env, 
                                               config_path="./neat-nsga2-config.init",
                                               scenario_ids=range(1,2),
                                               save_path="direct-neat",
                                               nsga2_use=True)

        # Train for 50 generations on scenarios 0-9
        policy.train(n_generations=100)

        # # Evaluate trained policy on scenario 0
        with WaterChlorinationEnv(**load_scenario(scenario_id=0)) as test_env:
            score = evaluate(policy, test_env)
            print("Evaluation metrics:", score)
