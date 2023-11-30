import openmdao.api as om

from aviary.mission.flops_based.ode.altitude_rate import AltitudeRate
from aviary.mission.flops_based.ode.range_rate import RangeRate
from aviary.mission.flops_based.ode.specific_energy_rate import SpecificEnergyRate
from aviary.variable_info.variables import Dynamic
from aviary.variable_info.enums import AnalysisScheme


class MissionEOM(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')
        self.options.declare("analysis_scheme", types=AnalysisScheme, default=AnalysisScheme.COLLOCATION,
                             desc="The analysis method that will be used to close the trajectory; for example collocation or time integration")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='specific_power',
                           subsys=SpecificEnergyRate(num_nodes=nn),
                           promotes_inputs=[Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS,
                                            Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.DRAG],
                           promotes_outputs=[Dynamic.Mission.SPECIFIC_ENERGY_RATE])
        self.add_subsystem(name=Dynamic.Mission.ALTITUDE_RATE,
                           subsys=AltitudeRate(num_nodes=nn),
                           promotes_inputs=[Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                                            Dynamic.Mission.VELOCITY_RATE, Dynamic.Mission.VELOCITY],
                           promotes_outputs=[Dynamic.Mission.ALTITUDE_RATE])
        self.add_subsystem(name='groundspeed',
                           subsys=RangeRate(num_nodes=nn),
                           promotes_inputs=[
                               Dynamic.Mission.ALTITUDE_RATE, Dynamic.Mission.VELOCITY],
                           promotes_outputs=[Dynamic.Mission.RANGE_RATE])
        self.add_subsystem(name='excess_specific_power',
                           subsys=SpecificEnergyRate(num_nodes=nn),
                           promotes_inputs=[(Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.THRUST_MAX_TOTAL),
                                            Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS, Dynamic.Mission.DRAG],
                           promotes_outputs=[(Dynamic.Mission.SPECIFIC_ENERGY_RATE, Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS)])
        self.add_subsystem(
            name=Dynamic.Mission.ALTITUDE_RATE_MAX,
            subsys=AltitudeRate(
                num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                 Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS),
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.VELOCITY],
            promotes_outputs=[
                (Dynamic.Mission.ALTITUDE_RATE,
                 Dynamic.Mission.ALTITUDE_RATE_MAX)])
