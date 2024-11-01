import openmdao.api as om

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp # was atmos_1976_corr
from openmdao.utils.units import convert_units
import numpy as np

from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


def units_setter(opt_meta, value):
    """
    Check and convert new units tuple into

    Parameters
    ----------
    opt_meta : dict
        Dictionary of entries for the option.
    value : any
        New value for the option.

    Returns
    -------
    any
        Post processed value to set into the option.
    """
    new_val, new_units = value
    old_val, units = opt_meta['val']

    converted_val = convert_units(new_val, new_units, units)
    return (converted_val, units)


class Atmosphere(om.Group):
    """
    Group that contains atmospheric conditions for the aircraft's current flight
    condition, as well as conversions for different speed types (TAS, EAS, Mach)
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )

        self.options.declare(
            'h_def',
            values=('geopotential', 'geodetic'),
            default='geopotential',
            desc='The definition of altitude provided as input to the component. If '
            '"geodetic", it will be converted to geopotential based on Equation 19 in '
            'the original standard.',
        )

        self.options.declare(
            'output_dsos_dh',
            types=bool,
            default=False,
            desc='If true, the derivative of the speed of sound will be added as an '
            'output',
        )

        self.options.declare(
            'isa_delta_temp',
            #types=tuple,
            default=(0.0, 'degR'),
            desc='Temperature delta (deg R) from typical International Standard Atmosphere (ISA) conditions',
            set_function=units_setter
        )

        self.options.declare(
            "input_speed_type",
            default=SpeedType.TAS,
            types=SpeedType,
            desc='defines input airspeed as equivalent airspeed, true airspeed, or mach '
            'number',
        )

    def setup(self):
        nn = self.options['num_nodes']
        speed_type = self.options['input_speed_type']
        h_def = self.options['h_def']
        output_dsos_dh = self.options['output_dsos_dh']
        isa_deltaT, _ = self.options['isa_delta_temp']

        self.add_subsystem(
            name='standard_atmosphere',
            subsys=USatm1976Comp(
                num_nodes=nn, h_def=h_def, output_dsos_dh=output_dsos_dh, #, isa_deltaT=isa_deltaT
            ),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                '*',
                #('sos', Dynamic.Mission.SPEED_OF_SOUND),
                #('rho', Dynamic.Mission.DENSITY),
                #('temp', Dynamic.Mission.TEMPERATURE),
                #('pres', Dynamic.Mission.STATIC_PRESSURE),
            ],
        )

        self.add_subsystem(
            name='standard_atmosphere_corrected',
            subsys=om.ExecComp([
                'sos_corr = sos * ((temp + temp_delta) / temp) ** 0.5',
                'rho_corr = rho * (temp / (temp + temp_delta))',
                'temp_corr = temp + temp_delta',
                'pres_corr = pres',
                'viscosity_corr = viscosity * ((temp + 198.72) / (temp + temp_delta + 198.72)) * ((temp + temp_delta) / temp) ** 1.5',
                'drhos_dh_corr = drhos_dh * (temp / (temp + temp_delta))',
                'dsos_dh_corr = dsos_dh * (temp / (temp + temp_delta)) ** 0.5'],
                temp_delta={'val': np.full(nn, isa_deltaT), 'units': 'degR'},
                sos={'shape': nn, 'units': 'ft/s'},
                rho={'shape': nn, 'units': 'slug/ft**3'},
                temp={'shape': nn, 'units': 'degR'},
                pres={'shape': nn, 'units': 'psi'},
                viscosity={'shape': nn, 'units': 'lbf*s/ft**2'},
                drhos_dh={'shape': nn, 'units': 'slug/ft**4'},
                dsos_dh={'shape': nn, 'units': '1/s'},
                sos_corr={'shape': nn, 'units': 'ft/s'},
                rho_corr={'shape': nn, 'units': 'slug/ft**3'},
                temp_corr={'shape': nn, 'units': 'degR'},
                pres_corr={'shape': nn, 'units': 'psi'},
                viscosity_corr={'shape': nn, 'units': 'lbf*s/ft**2'},
                drhos_dh_corr={'shape': nn, 'units': 'slug/ft**4'},
                dsos_dh_corr={'shape': nn, 'units': '1/s'},
                has_diag_partials=True,
            ),
            promotes_inputs=['*'],
            promotes_outputs=[
                ('sos_corr', Dynamic.Mission.SPEED_OF_SOUND),
                ('rho_corr', Dynamic.Mission.DENSITY),
                ('temp_corr', Dynamic.Mission.TEMPERATURE),
                ('pres_corr', Dynamic.Mission.STATIC_PRESSURE),
                'viscosity_corr',
                'drhos_dh_corr',
                'dsos_dh_corr',
            ],
        )

        self.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(num_nodes=nn, input_speed_type=speed_type),
            promotes=['*'],
        )
