
import numpy as np
from pydrake.systems.framework import BasicVector, LeafSystem
from pydrake.common.value import AbstractValue
from pydrake.multibody.plant import ExternallyAppliedSpatialForce
from pydrake.multibody.math import SpatialForce

class CartSystem(LeafSystem):

    def __init__(self, plant):
        LeafSystem.__init__(self)
        self.nv = plant.num_velocities()

        self.plant = plant

        self.cart_body = plant.GetBodyByName("slider_link")
        self.upper_link_body = plant.GetBodyByName("upper_arm_link")
        self.lower_link_body = plant.GetBodyByName("lower_arm_link")

        self.cart_body_index = self.cart_body.index()
        self.pend1_body_index = self.upper_link_body.index()
        self.pend2_body_index = self.lower_link_body.index()

        self.cart_joint = plant.GetJointByName("slider_joint")
        self.pend1_joint = plant.GetJointByName("shoulder_joint")
        self.pend2_joint = plant.GetJointByName("elbow_joint")

        self.root_context = plant.CreateDefaultContext()
        self.cart_kinematic = np.array([0., 0., 0., 0., 0., 0.])

        self.drag_coefficient = 0.05
        self.cart_mass = self.cart_body.default_mass()
        self.pend1_mass = self.upper_link_body.default_mass()
        self.pend2_mass = self.lower_link_body.default_mass()

        self.DeclareAbstractOutputPort(
            "spatial_forces_vector",
            lambda: AbstractValue.Make(
            [ExternallyAppliedSpatialForce()]*3), self.DoCalcAbstractOutput)

        self.DeclareVectorOutputPort(
            "generalized_forces",
            BasicVector(self.nv),
            self.DoCalcVectorOutput)
    
    def DoCalcAbstractOutput(self, context, y_data):
        forces = []

        cart_force = ExternallyAppliedSpatialForce()
        cart_force.body_index = self.cart_body_index
        cart_force.p_BoBq_B = np.zeros(3)
        cart_force.F_Bq_W = SpatialForce(tau=self.cart_kinematic[3:], 
                                         f=self.cart_kinematic[:3])
        forces.append(cart_force)

        plant_context = self.plant.GetMyContextFromRoot(self.root_context)
        spatial_vel_upper = self.plant.EvalBodySpatialVelocityInWorld(plant_context, self.upper_link_body)
        spatial_vel_lower = self.plant.EvalBodySpatialVelocityInWorld(plant_context, self.lower_link_body)
  
        drag_force_upper_rotational = self.CalculateDragForceRotational(spatial_vel_upper, self.drag_coefficient)
        drag_force_upper_translational = self.CalculateDragForceTranslational(spatial_vel_upper, self.drag_coefficient)
        drag_force_lower_rotational = self.CalculateDragForceRotational(spatial_vel_lower, self.drag_coefficient)
        drag_force_lower_translational = self.CalculateDragForceTranslational(spatial_vel_lower, self.drag_coefficient)

        pend1_force = ExternallyAppliedSpatialForce()
        pend1_force.body_index = self.pend1_body_index
        pend1_force.p_BoBq_B = np.zeros(3)
        pend1_force.F_Bq_W = SpatialForce(tau=drag_force_upper_rotational, f=drag_force_upper_translational)
        forces.append(pend1_force)

        pend2_force = ExternallyAppliedSpatialForce()
        pend2_force.body_index = self.pend2_body_index
        pend2_force.p_BoBq_B = np.zeros(3)
        pend2_force.F_Bq_W = SpatialForce(tau=drag_force_lower_rotational, f=drag_force_lower_translational)
        forces.append(pend2_force)

        y_data.set_value(forces)

    def DoCalcVectorOutput(self, context, y_data):
        y_data.SetFromVector(np.zeros(self.nv))

    def CalculateDragForceRotational(self, spatial_velocity, drag_coefficient):
        velocity = spatial_velocity.rotational()
        vel_mag = np.linalg.norm(velocity)
        area_cylinder = 2 * 0.01 * 1
        air_density = 1225

        if vel_mag > 1e-6:
            drag_force = -(0.5) * drag_coefficient * air_density * area_cylinder * velocity
        else:
            drag_force = np.zeros(3)
        return drag_force
    
    def CalculateDragForceTranslational(self, spatial_velocity, drag_coefficient):
        velocity = spatial_velocity.translational()
        vel_mag = np.linalg.norm(velocity)
        area_cylinder = 2 * 0.01 * 1
        air_density = 1225

        if vel_mag > 1e-6:
            drag_force = -(0.5) * drag_coefficient * air_density * area_cylinder * velocity
        else:
            drag_force = np.zeros(3)
        return drag_force