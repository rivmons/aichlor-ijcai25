"""
This module contains the water chlorination control environment that is to be used in the
"1st AI for Drinking Water Chlorination Challenge" @ IJCAI-2025.
"""
from typing import Optional, Any
import numpy as np
from epyt_control.envs import EpanetMsxControlEnv
from epyt_control.envs.actions import SpeciesInjectionAction
from epyt_flow.simulation import ScadaData, SensorConfig, ScenarioConfig
from epyt_flow.utils import to_seconds


class WaterChlorinationEnv(EpanetMsxControlEnv):
    """
    Control environment.
    """
    def __init__(self, scenario_config: ScenarioConfig, f_in_contamination_metadata: str,
                 f_in_streams_data: str, action_space: list[SpeciesInjectionAction],
                 f_hyd_file_in: str = None, hyd_scada_in: ScadaData = None):
        super().__init__(scenario_config=scenario_config,
                         action_space=action_space,
                         rerun_hydraulics_when_reset=False,
                         hyd_file_in=f_hyd_file_in, hyd_scada_in=hyd_scada_in,
                         reload_scenario_when_reset=False)

        self.__sensor_config_reward = None
        self._f_in_contamination_metadata = f_in_contamination_metadata
        self._f_in_streams_data = f_in_streams_data

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
              ) -> tuple[np.ndarray, dict]:
        # Reset
        super().reset(seed, options)

        # Set constant chlorine injection
        #self._scenario_sim.epanet_api.setMSXPattern("CL2PAT", [3000])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT1", [500])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT2", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT3", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT4", [10])
        self._scenario_sim.epanet_api.setMSXPattern("CL2PAT5", [10])

        # Skip first three days to give the network time to settle a proper initial state
        time_step = self._scenario_sim.epanet_api.getTimeHydraulicStep()
        n_steps_to_skip = int(to_seconds(days=3) / time_step)

        current_scada_data = None
        for _ in range(n_steps_to_skip):
            current_scada_data, _ = self._next_sim_itr()

        obs = self._get_observation(current_scada_data)

        return obs, {"scada_data": current_scada_data}

    def reward_0(self, scada_data: ScadaData) -> float:
        """
        Computes the current reward based on the current sensors readings (i.e. SCADA data).
        Sums up (negative) residuals for out of bounds Cl concentrations at nodes -- i.e.
        reward of zero means everythings is okay, while a negative reward denotes Cl concentration
        bound violations

        Parameters
        ----------
        :class:`epyt_flow.simulation.ScadaData`
            Current sensor readings.

        Returns
        -------
        `float`
            Current reward.
        """
        reward = 0.

        # Regulation Limits (taken from the evaluation metrics)
        upper_cl_bound = .4  # (mg/l)
        lower_cl_bound = .2  # (mg/l)

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

        upper_bound_violation_idx = nodes_quality > upper_cl_bound
        reward += -1. * np.sum(nodes_quality[upper_bound_violation_idx] - upper_cl_bound)

        lower_bound_violation_idx = nodes_quality < lower_cl_bound
        reward += np.sum(nodes_quality[lower_bound_violation_idx] - lower_cl_bound)

        return reward
    
    def reward_1(self, scada_data: ScadaData) -> float:
        upper_cl_bound = 0.4
        lower_cl_bound = 0.2

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})

        over = np.clip(nodes_quality - upper_cl_bound, 0, None)
        under = np.clip(lower_cl_bound - nodes_quality, 0, None)

        cl_penalty = -np.mean(over + under)

        control_cost = -0.01 * np.sum(scada_data.get_data_control())  # scale this down

        injection = scada_data.get_data_control()
        if injection.shape[0] > 1:
            smoothness_penalty = -0.1 * np.mean(np.abs(np.diff(injection, axis=0)))
        else:
            smoothness_penalty = 0.0

        reward = cl_penalty + control_cost + smoothness_penalty
        return reward

    def _compute_reward_function(self, scada_data: ScadaData, action: np.ndarray) -> float:
        upper_cl_bound = 0.4
        lower_cl_bound = 0.2

        if self.__sensor_config_reward is None:
            self.__sensor_config_reward = SensorConfig.create_empty_sensor_config(scada_data.sensor_config)
            self.__sensor_config_reward.bulk_species_node_sensors = {"CL2": scada_data.sensor_config.nodes}
        scada_data.change_sensor_config(self.__sensor_config_reward)

        nodes_quality = scada_data.get_data_bulk_species_node_concentration({"CL2": scada_data.sensor_config.nodes})
        action = action.flatten()

        over = np.clip(nodes_quality - upper_cl_bound, 0, None)
        under = np.clip(lower_cl_bound - nodes_quality, 0, None)

        cl_penalty = -np.mean(over + under) * 10

        violations_per_node = np.mean(over + under, axis=0)
        fairness_penalty = -(np.max(violations_per_node) - np.min(violations_per_node))

        control_cost = -0.1 * np.sum(action)

        if action.ndim > 1 and action.shape[0] > 1:
            smoothness_penalty = -0.1 * np.mean(np.abs(np.diff(action, axis=0)))
        else:
            smoothness_penalty = 0.0

        if hasattr(scada_data, "get_data_bulk_species_node_concentration"):
            try:
                pathogen_conc = scada_data.get_data_bulk_species_node_concentration({"P": scada_data.sensor_config.nodes})
                infection_risk_penalty = -0.5 * np.mean(pathogen_conc)  # scale weight as needed
            except Exception:
                infection_risk_penalty = 0.0
        else:
            infection_risk_penalty = 0.0

        reward = (cl_penalty
                + fairness_penalty
                + control_cost
                + smoothness_penalty
                + infection_risk_penalty)
        
        scale = 10.0
        scaled_reward = reward * scale

        reward_info = {
            "cl_penalty": cl_penalty,
            "fairness_penalty": fairness_penalty,
            "control_cost": control_cost,
            "smoothness_penalty": smoothness_penalty,
            "infection_risk_penalty": infection_risk_penalty,
            "cl2_violation_upper": float(np.mean(over)),
            "cl2_violation_lower": float(np.mean(under)),
            "cl2_total_deviation": float(np.mean(over + under)),
            "scaled_reward": scaled_reward
        }

        oneobj_simpler = -np.mean(np.clip(lower_cl_bound - nodes_quality, 0, None)) * 10

        return oneobj_simpler, reward_info


