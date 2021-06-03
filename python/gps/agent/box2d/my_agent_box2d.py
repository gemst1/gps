""" This file defines an agent for the Box2D simulator. """
from copy import deepcopy
import numpy as np
from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import myAGENT_BOX2D
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import JOINT_ANGLES

import rospy
from std_msgs.msg import Float64

class AgentBox2D(Agent):
    """
    All communication between the algorithms and Box2D is done through
    this class.
    """
    def __init__(self, hyperparams):
        config = deepcopy(myAGENT_BOX2D)
        config.update(hyperparams)
        Agent.__init__(self, config)

        rospy.init_node('dummy_agent', anonymous=True)
        self.pub = rospy.Publisher('des_dis', Float64, queue_size=1)
        self.sub = rospy.Subscriber('des_dis', Float64, self.callback)

        self.x0 = self._hyperparams["x0"]

    def callback(self, distance):
        dis = distance.data
        self.sub_state = {JOINT_ANGLES: np.array([dis])}
        self.waiting = False

    def pub_and_sub(self, msg_to_pub):
        self._waiting = True
        self.pub.publish(msg_to_pub)

        time_waited = 0
        while self._waiting:
            rospy.sleep(0.1)
            time_waited += 0.01
            if time_waited > 0.5:
                raise TimeoutException(time_waited)
        return self.sub_state


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
        b2d_X = self.pub_and_sub(0)
        new_sample = self._init_sample(b2d_X)
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
                b2d_X = self.pub_and_sub(U[t, :])
                self._set_sample(new_sample, b2d_X, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        self.pub.publish(5)
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
