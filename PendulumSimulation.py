

import torch
import numpy as np

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.planar_scenegraph_visualizer import (
    ConnectPlanarSceneGraphVisualizer)

from CartSystem import CartSystem

class PendulumSimulation:
    def __init__(self, playback=True, show=True, enable_visualizer=True):
        """
        Initializes the pendulum simulation.

        Arguments:
            playback: Enable pyplot animations to be produced.
            show: Show the visualizer.
            enable_visualizer: Enable the visualizer.
        """

        self.playback = playback
        self.show = show
        self.enable_visualizer = enable_visualizer

        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(self.builder, 0.)
        self.parser = Parser(self.plant)
        self.parser.AddModels("./double_pendulum_free.urdf")

        self.plant.Finalize()

        if self.enable_visualizer:
            T_VW = np.array([[1., 0., 0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
            self.visualizer = ConnectPlanarSceneGraphVisualizer(
                self.builder, self.scene_graph, T_VW=T_VW, xlim=[-15.5, 15.5],
                ylim=[-6.5, 6.5], show=self.show)
            if self.playback:
                self.visualizer.start_recording()

        self.cart_system = self.builder.AddSystem(CartSystem(self.plant))

        self.builder.Connect(
            self.cart_system.GetOutputPort("spatial_forces_vector"),
            self.plant.GetInputPort("applied_spatial_force")
        )
        self.builder.Connect(
            self.cart_system.GetOutputPort("generalized_forces"),
            self.plant.GetInputPort("applied_generalized_force")
        )

        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.Initialize()

        if self.enable_visualizer:
            self.simulator.set_target_realtime_rate(1.)

        self.cart_body = self.cart_system.cart_body
        cart_joint = self.cart_system.cart_joint
        pend1_joint = self.cart_system.pend1_joint
        pend2_joint = self.cart_system.pend2_joint

        self.cart_mass = self.cart_body.default_mass()

        self.cart_system.root_context = self.simulator.get_context()
        self.cart_system.plant = self.plant

    def reset_simulation(self):
        """
        Resets the simulation to its initial state.
        """
        reset_context = self.simulator.get_system().CreateDefaultContext()
        self.cart_system.root_context = reset_context
        self.simulator.reset_context(self.cart_system.root_context)
        self.simulator.Initialize()

    def normalize_cart_pos(self, cart_pos, max_pos=5):
        return cart_pos / max_pos
    
    def normalize_angular_velocity(self, w, max_w=10):
        return w / max_w
    
    def normalize_angle(self, angle):
        norm_angle = angle % (2 * np.pi)
        dist_from_pi = norm_angle - np.pi
        if dist_from_pi > np.pi:
            dist_from_pi -= 2 * np.pi
        if dist_from_pi < -np.pi:
            dist_from_pi += 2 * np.pi
        return dist_from_pi
    
    def pendulum_angular(self, pend1_joint, pend2_joint, context):  
        pend1_w = pend1_joint.get_angular_rate(context)
        pend1_theta = pend1_joint.get_angle(context)
        pend2_w = pend2_joint.get_angular_rate(context)
        absolute_pend2_w = pend1_w + pend2_w
        pend2_theta = pend2_joint.get_angle(context)
        absolute_pend2_theta = pend1_theta + pend2_theta
        return pend1_w, pend1_theta, absolute_pend2_w, absolute_pend2_theta
    
    def pendulum_positions(self, pend1_theta, pend2_theta):
        pend1_x = -np.sin(pend1_theta)
        pend1_y = -np.cos(pend1_theta)
        pend2_x = pend1_x - np.sin(pend2_theta)
        pend2_y = pend1_y - np.cos(pend2_theta)
        return pend1_x, pend1_y, pend2_x, pend2_y

    def run(self, model, duration=60, time_step=0.1, gravity=9.81, drag_coefficient=0.05, acceleration=10.):
        self.reset_simulation()
        # Initialize the cart system
        self.cart_system.drag_coefficient = drag_coefficient
        self.plant.mutable_gravity_field().set_gravity_vector(np.array([0., 0., -gravity]))
        pend1_x_prev, pend1_y_prev = 0., 0.
        pend2_x_prev, pend2_y_prev = 0., 0.

        pendulum2_heights = []
        outputs = []

        cart_travelled = 0
        time_over_threshold = 0
        best_time_over_threshold = 0

        cart_body = self.cart_system.cart_body
        cart_joint = self.cart_system.cart_joint
        pend1_joint = self.cart_system.pend1_joint
        pend2_joint = self.cart_system.pend2_joint
        cart_mass = self.cart_system.cart_mass
        score = 0
        counter = 0

        while self.simulator.get_context().get_time() < duration:
            curr_context = self.diagram.GetMutableSubsystemContext(self.plant, self.simulator.get_mutable_context())
            cart_velocity = self.plant.GetVelocities(curr_context)[0]

            cart_pos = cart_joint.GetOnePosition(curr_context)
            
            if self.simulator.get_context().get_time() > 0:
                cart_travelled += abs(cart_pos - cart_pos_prev)

            pend1_w, pend1_theta, pend2_w, pend2_theta = self.pendulum_angular(pend1_joint, pend2_joint, curr_context)
            pend1_x, pend1_y, pend2_x, pend2_y = self.pendulum_positions(pend1_theta, pend2_theta)
            n_pend1_theta = self.normalize_angle(pend1_theta)
            n_pend2_theta = self.normalize_angle(pend2_theta)

            pendulum2_heights.append(pend2_y)

            if self.simulator.get_context().get_time() >= time_step:
                pend1_vector = [pend1_x - pend1_x_prev, pend1_y - pend1_y_prev]
                pend2_vector = [pend2_x - pend2_x_prev, pend2_y - pend2_y_prev]
            else:
                pend1_vector = pend2_vector = [0., 0.]

            pend_dot = np.dot(pend1_vector, pend2_vector)

            # Query the model for the cart acceleration
            data = np.array([[cart_pos, cart_velocity, pend1_w, n_pend1_theta, pend2_w, n_pend2_theta, pend_dot]])
            inputs = torch.Tensor(data)
            cart_acceleration = model(inputs).item() * 100
            outputs.append(cart_acceleration)

            force_value = cart_acceleration * cart_mass
            self.cart_system.cart_kinematic = np.array([force_value, 0., 0., 0., 0., 0.])

            # Check if the pendulum is over the threshold
            if pend2_y > 1.9:
                time_over_threshold += time_step
                counter += time_step
            else:
                if counter > best_time_over_threshold:
                    best_time_over_threshold = counter
                counter = 0

            # Update the previous values
            cart_pos_prev = cart_pos
            pend1_x_prev, pend1_y_prev = pend1_x, pend1_y
            pend2_x_prev, pend2_y_prev = pend2_x, pend2_y

            if abs(cart_pos) > 10:
                break

            if self.simulator.get_context().get_time() > 5 and time_over_threshold == 0:
                break

            # Advance the simulation
            self.simulator.AdvanceTo(self.simulator.get_context().get_time() + time_step)

        final_time = self.simulator.get_context().get_time()

        score = (100*best_time_over_threshold) / (1 + cart_travelled)

        if self.playback and self.enable_visualizer:
            self.visualizer.stop_recording()
            ani = self.visualizer.get_recording_as_animation()
            return ani
        else:
            return model, score, best_time_over_threshold, final_time