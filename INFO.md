### `conda env create -f environment.yml` should work.
- tested on ubuntu linux machine
- only thing weird i saw is that some versions of dependencies make epyt throw a weird error where "topologies don't match"
- shouldn't happen in this case

### you will need to change one file in epyt_control: `rl_env.py`
* error you'll get: 
`Traceback (most recent call last):
  File "/home/rmonsia/test/example.py", line 24, in <module>
    r = evaluate(my_policy, env)
  File "/home/rmonsia/test/evaluation.py", line 43, in evaluate
    obs, reward, terminated, _, info = env.step(action)
  File "/home/rmonsia/miniforge3/envs/aichlor/lib/python3.9/site-packages/epyt_control/envs/rl_env.py", line 252, in step
    current_reward = self._compute_reward_function(deepcopy(current_scada_data))
TypeError: _compute_reward_function() missing 1 required positional argument: 'action'`
* **go to the rl_env.py file and just copy and paste './rl_env.txt' to that file**

### important parts of code:
#### scenario_ids = [0...10] (10 is 365 day scenario)
- reward function in env.py (_compute_reward_function(...))
- normal neat policy in control_policy.py
- predictor logic in predictor.py
- evaluation function (end-of-scenario 5 objs) in evaluation.py
- pipeline for train predictor -> train prescriptors -> collect data in pipeline.py
- ppo training in ppo.py (with tensorboard and a bunch of logging i added) in sb3.py
- nsga2 implementation in nsga2.py

# There's 3 different things you could do: 
### train neat prescriptors directly via nsga2 and evalution metrics 
* should just work with `python example.py`
* you can change scenario_ids in policy declaration to change with scenarios to train on
* nsga_use=True -> use nsga2 (i think i hard-coded to only work with nsga2 right now anyway?)
* saves to save_path
* need to pass dummy env; isn't actually used
* **policy.train(n_generations=)** to train
* config is neat-nsga2-config.init


### train via pipeline (olivier's)
* should just work with `python pipeline.py`
* can change number of epochs for predictor training (line 242), num gens for prescriptor training (line 267)
* outputs predictor scaler and model to './' as reward_scaler.pkl and model.weights.h5
* outputs neat populations for training to './neat'
* outputs data to 'data/it_{i}' given iteration of cycle
* **need data/it_0 before running as initial data**


### train a ppo (mlplstm) policy
#### i think you'll need to change library-specific code for epyt-control if you use 
* should just work with `python sb3.py`
* logs to tensorboard
* collects data to logs/surrogate_training_data.csv that you can just move to data/it_{} directly without any processing



## what to do (in no particular order)?
### 1) collect better data for data/it_0
### 2) compare direct evol with simple reward with surrogate-assisted (performance and time)
### 3) compare composite reward fn with nsga2 when evolving (surrogate and none)
### 4) curricular learning with ppo to make sure reward function good enough to use with ESP (and inject data to train predictor)
        * apply same curricular learning to ESP setup?
### 5) get a run via ESP with better reward fn