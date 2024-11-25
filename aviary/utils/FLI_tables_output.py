'''--------------------------------------------------------------------------------------------
Aviary timeseries output function that generates FLIXXX and FLIPATH tables for ANOPP/ANOPP2.


FLIPATH:

Description                     Aviary Variable                     Format          Units
-----------------------------------------------------------------------------------------------
Time                            time                                XXXXX.XX        s
Aircraft x position             distance                            XXXXXXX.XX      ft
Aircraft y position             none (lateral) [=0]                 X.XX            ft
Aircraft z position             - altitude                          XXXXXX.X        ft
PsiB (1st EFB Euler angle)      none (psiB) [=0]                    X.XX            deg
ThetaB (2nd EFB Euler angle)    Mission.Takeoff.THRUST_INCIDENCE    XXX.XX          deg
                                + flight_path_angle 
                                + alpha
PhiB (3rd EFB Euler angle)      none (phiB) [=0]                    X.XX            deg
PsiWB (1st WB Euler angle)      none (psiWB) [=0]                   X.XX            deg
ThetaWB (2nd WB Euler angle)    Mission.Takeoff.THRUST_INCIDENCE    XXX.XX          deg
                                + alpha
PhiWB (3rd WB Euler angle)      none (phiWB) [=0]                   X.XX$           deg


FLIXXX:

Description                     Aviary Variable                     Format          Units
-----------------------------------------------------------------------------------------------
Time                            time                                XXXXX.XX        s
Flight Mach number              mach                                X.XXXX
Power setting                   throttle                            X.XXX
Sound speed                     speed_of_sound                      XXXX.X          ft/s
Density                         density                             X.XXXXXXXX      slug/ft^3
Dynamic viscosity               viscosity                           X.XXXXXXXXXXXX  slug/(ft*s)
Gear indicator                  none (gear) [=arg]                  XXXXXX          
Flap deflection                 none (flap) [=arg]                  XX.X            deg
Absolute humidity               abs_humidity                        X.XXX$          %mol.frac.

--------------------------------------------------------------------------------------------'''

import numpy as np
import pandas as pd

#from pathlib import Path
#from aviary.interface.methods_for_level2 import wrapped_convert_units


def timeseries_output(prob, path, phase_gearup, phase_flapsup, angle_flaps, angle_thrust):

    output_names = ['time', 'distance', 'lateral', 'altitude', 'psiB', 'thetaB', 'phiB', 'psiWB', 'thetaWB', 'phiWB', 'flight_path_angle', 'alpha', 'mach', 
                   'throttle', 'speed_of_sound', 'density', 'viscosity', 'gear', 'flap', 'abs_humidity'] 
    
    # Data handling
    phase_names = list(prob.model.traj._phases.keys())
    timeseries_names = [f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}' for phase_name in phase_names for output_name in output_names]
    timeseries_dict = prob.model.list_outputs(includes=timeseries_names, out_stream=None, return_format='dict', units=True)
    for key in timeseries_dict.keys():
        timeseries_dict[key]['val'] = timeseries_dict[key]['val'].flatten()

    FLIPATH_list = [[] for i in range(10)]
    FLIXXX_list = [[] for i in range(9)]
    for phase_name in phase_names:
        for output_name in output_names:

            # Fill in additional timeseries data
            if f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}' not in timeseries_dict.keys():

                if output_name == 'thetaB':
                    thetaB = timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.alpha']['val'] + angle_thrust
                    if f'traj.phases.{phase_name}.timeseries.timeseries_comp.flight_path_angle' in timeseries_dict.keys():
                        thetaB += timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.flight_path_angle']['val']

                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'deg',
                        'val': thetaB,
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }
                
                elif output_name == 'thetaWB':
                    thetaWB = timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.alpha']['val'] + angle_thrust

                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'deg',
                        'val': thetaWB,
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }
                
                elif output_name == 'throttle':
                    append_len = len(timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.time']['val'])
                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'unitless',
                        'val': np.ones(append_len),
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }

                elif output_name == 'gear':
                    append_len = len(timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.time']['val'])
                    gearup_index = phase_names.index(phase_gearup) if phase_gearup in phase_names else 1000
                    if phase_names.index(phase_name) >= gearup_index:
                        gear = np.full(append_len, '4HUP  ')
                    else:
                        gear = np.full(append_len, '4HDOWN')

                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'unitless',
                        'val': gear,
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }
                
                elif output_name == 'flap':
                    append_len = len(timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.time']['val'])
                    flapsup_index = phase_names.index(phase_flapsup) if phase_flapsup in phase_names else 1000
                    if phase_names.index(phase_name) >= flapsup_index:
                        flap = np.zeros(append_len)
                    else:
                        flap = np.full(append_len, angle_flaps)

                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'unitless',
                        'val': flap,
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }

                else:
                    append_len = len(timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.time']['val'])
                    timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}'] = {
                        'units': 'unitless',
                        'val': np.zeros(append_len),
                        'prom_name': f'traj.{phase_name}.timeseries.{output_name}',
                    }
            
            # Create FLIPATH table
            for n in timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}']['val']:
                if output_name == 'time':
                    str_list = [str(f'{n:.2f}').rjust(8)]
                    FLIPATH_list[0].extend(str_list)
                elif output_name == 'distance':
                    str_list = [str(f'{n:.2f}').rjust(10)]
                    FLIPATH_list[1].extend(str_list)
                elif output_name == 'lateral':
                    str_list = [str(f'{n:.2f}').rjust(4)]
                    FLIPATH_list[2].extend(str_list)
                elif output_name == 'altitude':
                    str_list = [str(f'{-n:.1f}').rjust(8)] # format as negative
                    FLIPATH_list[3].extend(str_list)
                elif output_name == 'psiB':
                    str_list = [str(f'{n:.2f}').rjust(4)]
                    FLIPATH_list[4].extend(str_list)
                elif output_name == 'thetaB':
                    str_list = [str(f'{n:.2f}').rjust(6)]
                    FLIPATH_list[5].extend(str_list)
                elif output_name == 'phiB':
                    str_list = [str(f'{n:.2f}').rjust(4)]
                    FLIPATH_list[6].extend(str_list)
                elif output_name == 'psiWB':
                    str_list = [str(f'{n:.2f}').rjust(4)]
                    FLIPATH_list[7].extend(str_list)
                elif output_name == 'thetaWB':
                    str_list = [str(f'{-n:.2f}').rjust(6)] # format as negative
                    FLIPATH_list[8].extend(str_list)
                elif output_name == 'phiWB':
                    str_list = [str(f'{n:.2f}$').rjust(5)]
                    FLIPATH_list[9].extend(str_list)
            
            # Create FLIXXX table
            for n in timeseries_dict[f'traj.phases.{phase_name}.timeseries.timeseries_comp.{output_name}']['val']:
                if output_name == 'time':
                    str_list = [str(f'{n:.2f}').rjust(8)]
                    FLIXXX_list[0].extend(str_list)
                elif output_name == 'mach':
                    str_list = [str(f'{n:.4f}').rjust(6)]
                    FLIXXX_list[1].extend(str_list)
                elif output_name == 'throttle':
                    str_list = [str(f'{n*0.9:.3f}').rjust(5)] # format as derate
                    FLIXXX_list[2].extend(str_list)
                elif output_name == 'speed_of_sound':
                    str_list = [str(f'{n:.1f}').rjust(6)]
                    FLIXXX_list[3].extend(str_list)
                elif output_name == 'density':
                    str_list = [str(f'{n:.8f}').rjust(10)]
                    FLIXXX_list[4].extend(str_list)
                elif output_name == 'viscosity':
                    str_list = [str(f'{n:.12f}').rjust(14)]
                    FLIXXX_list[5].extend(str_list)
                elif output_name == 'gear':
                    str_list = [n]
                    FLIXXX_list[6].extend(str_list)
                elif output_name == 'flap':
                    str_list = [str(f'{n:.1f}').rjust(4)]
                    FLIXXX_list[7].extend(str_list)
                elif output_name == 'abs_humidity':
                    str_list = [str(f'{n:.3f}$').rjust(6)]
                    FLIXXX_list[8].extend(str_list)

    # Write tables to text files
    FLIPATH_table = pd.DataFrame(FLIPATH_list).transpose()
    FLIXXX_table = pd.DataFrame(FLIXXX_list).transpose()
    with open(f'{path}/FLIPATH.txt', mode='w') as file:
        for row in FLIPATH_table.itertuples(index=False):
            file.write(' '.join(map(str, row)) + '\n')
    with open(f'{path}/FLIXXX.txt', mode='w') as file:
        for row in FLIXXX_table.itertuples(index=False):
            file.write(' '.join(map(str, row)) + '\n')




