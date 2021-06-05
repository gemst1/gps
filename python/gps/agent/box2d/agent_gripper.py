""" This file defines an agent for the Box2D simulator. """
from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_GRIPPER
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import JOINT_ANGLES, END_EFFECTOR_POINTS
from gps.agent.ros.ros_utils import ServiceEmulator

import rospy
from std_msgs.msg import Float32

class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(AGENT_GRIPPER)
        config.update(hyperparams)
        Agent.__init__(self, config)

        rospy.init_node('gps_gripper_agent', anonymous=True)
        self._init_pubs_and_subs()

        self.x0 = self._hyperparams["x0"]

    def msg_to_state(self, msg):
        dis = msg.data
        state = {JOINT_ANGLES: np.array([dis, 0]), END_EFFECTOR_POINTS: np.array([dis, 0, 0])}
        return state

    def _init_pubs_and_subs(self):
        self._trial_service = ServiceEmulator(
            self._hyperparams['trial_command_topic'], Float32,
            self._hyperparams['state_result_topic'], Float32
        )
        self._reset_service = ServiceEmulator(
            self._hyperparams['reset_command_topic'], Float32,
            self._hyperparams['state_result_topic'], Float32
        )


    def sample(self, policy, condition, verbose=False, save=True, noisy=True):
        """
        Runs a trial and constructs a new sample containing information
        about the trial.

        Args:
            policy: Policy to to used in the trial.
            condition (int): Which condition setup to run.
            verbose (boolean): Whether or not to plot the trial (not used here).
            save (boolean): Whether or not to store the trial into the samples.
            noisy (boolean): Whether or not to use noise during sampling.
        """
        state_msg = self._reset_service.publish_and_wait(0)
        state = self.msg_to_state(state_msg)
        new_sample = self._init_sample(state)
        U = np.zeros([self.T, self.dU])
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))
        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
            if (t+1) < self.T:
                state_msg = self._trial_service.publish_and_wait(U[t, :])
                state = self.msg_to_state(state_msg)
                self._set_sample(new_sample, state, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, b2d_X):
        """
        Construct a new sample and fill in the first time step.
        """
        sample = Sample(self)
        self._set_sample(sample, b2d_X, -1)
        return sample

    def _set_sample(self, sample, b2d_X, t):
        for sensor in b2d_X.keys():
            sample.set(sensor, np.array(b2d_X[sensor]), t=t+1)
