
from copy import deepcopy

class ParamsHolder:
    def __init__(self):
        # Template parameters.

        self.config_dict = dict()
        self.config_dict['dt'] = 0.1

        # # Izhikevich's model
        self.config_dict['izhi_c1'] = 0.04
        self.config_dict['izhi_c2'] = 5
        self.config_dict['izhi_c3'] = 140
        self.config_dict['izhi_a_ex'] = 0.035
        self.config_dict['izhi_b_ex'] = 0.2
        self.config_dict['izhi_c_ex'] = -60
        self.config_dict['izhi_d_ex'] = 8
        self.config_dict['izhi_a_in'] = 0.02
        self.config_dict['izhi_b_in'] = 0.25
        self.config_dict['izhi_c_in'] = -65
        self.config_dict['izhi_d_in'] = 2
        self.config_dict['V_ex'] = 0
        self.config_dict['V_in'] = -80
        self.config_dict['V_thresh'] = 30
        self.config_dict['spdelay'] = int(2 / self.config_dict['dt'])  # unit = indices
        self.config_dict['noise_rate'] = 0

        # # Theta inhibition
        self.config_dict['theta_amp'] = 7
        self.config_dict['theta_f'] = 10

        # Positional drive
        self.config_dict['EC_phase_deg'] = 290
        self.config_dict['Ipos_max'] = 2  # Real value is Ipos_max + 4.7, compensated for EC phase during simulation.
        self.config_dict['Iangle_diff'] = 0
        self.config_dict['Iangle_compen'] = 0
        self.config_dict['Ipos_sd'] = 5
        self.config_dict['Iangle_kappa'] = 1
        self.config_dict['ECstf_rest'] = 1
        self.config_dict['ECstf_target'] = 1
        self.config_dict['tau_ECstf'] = 0.5e3
        self.config_dict['U_ECstf'] = 0.001

        # Sensory tuning
        self.config_dict['xmin'] = -40
        self.config_dict['xmax'] = 40
        self.config_dict['nx_ca3'] = 80
        self.config_dict['nx_mos'] = 0
        self.config_dict['ymin'] = -40
        self.config_dict['ymax'] = 40
        self.config_dict['ny_ca3'] = 80
        self.config_dict['ny_mos'] = 0
        self.config_dict['nn_inca3'] = 250
        self.config_dict['nn_inmos'] = 250

        # Synapse parameters
        self.config_dict['tau_gex'] = 12
        self.config_dict['tau_gin'] = 10
        self.config_dict['U_stdx_CA3'] = 0
        self.config_dict['U_stdx_mos'] = 0
        self.config_dict['tau_stdx'] = 0.5e3

        # # Weights
        # CA3-CA3
        self.config_dict['wmax_ca3ca3'] = 1000
        self.config_dict['wmax_ca3ca3_adiff'] = 0
        self.config_dict['w_ca3ca3_akappa'] = 2
        self.config_dict['asym_flag'] = True

        # CA3-Mos and Mos-CA3
        self.config_dict['wmax_ca3mos'] = 0
        self.config_dict['wmax_mosca3'] = 0
        self.config_dict['wmax_ca3mosca3_adiff'] = 0
        self.config_dict['mos_exist'] = False
        self.config_dict['w_ca3mosca3_akappa'] = 1

        # CA3-In and In-CA3
        self.config_dict['wmax_CA3in'] = 0
        self.config_dict['wmax_inCA3'] = 0

        # Mos-In and In-Mos
        self.config_dict['wmax_Mosin'] = 0
        self.config_dict['wmax_inMos'] = 0

        # Mos-Mos, In-In
        self.config_dict['wmax_mosmos'] = 0
        self.config_dict['wmax_inin'] = 0

        # SD
        self.config_dict['wsd_global'] = 2

        # ============ Defined in simulation script ========
        # Mossy layer projection trajectory
        self.config_dict['mos_startpos'] = 0
        self.config_dict['mos_endpos'] = 0
        # ===================================================


    def fig1_in(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['wmax_ca3ca3'] = 1100
        config_dict['Ipos_max'] = 3
        return config_dict

    def fig1_ex(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3ca3'] = 1100  # 1000
        config_dict['U_stdx_CA3'] = 0.9
        config_dict['Ipos_max'] = 4.5
        return config_dict

    def fig2(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Iangle_diff'] = 6
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 4
        config_dict['U_stdx_CA3'] = 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 2000
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        return config_dict

    def fig3(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Iangle_diff'] = 6
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 1500
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 3000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict


    def fig3_Speed(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['dt'] = 0.1
        config_dict['spdelay'] = int(2 / config_dict['dt'])  # unit = indices
        config_dict['Ipos_max'] = 2  # 2
        config_dict['Ipos_sd'] = 15  # 5
        config_dict['Iangle_diff'] = 6  # 6
        config_dict['wsd_global'] = 2  # 2
        config_dict['tau_ECstf'] = 0.5e3  # 0.5e3
        config_dict['tau_stdx'] = 0.5e3  # 0.5e3
        config_dict['wmax_ca3ca3_adiff'] = 1500  # 1500
        config_dict['wmax_ca3mosca3_adiff'] = 3000  # 3000
        config_dict['theta_f'] = 10  # 10
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        # ========================================================
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40

        config_dict['wmax_ca3ca3'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False

        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict

    def fig4_Ctrl(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Iangle_diff'] = 6
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 2000
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 3000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict

    def fig4_DGlesion(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Iangle_diff'] = 6
        config_dict['ECstf_rest'] = 1.25
        config_dict['ECstf_target'] = 1.25
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['U_ECstf'] = 0
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 2000
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 0
        config_dict['mos_exist'] = False
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict

    def fig5(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Iangle_diff'] = 6
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 2000
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 4000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict


    def fig6(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 2
        config_dict['Iangle_diff'] = 6
        config_dict['Iangle_compen'] = 2
        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 1500
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 4000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict


    def fig2_NoRecurrence(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 5 #6
        config_dict['Iangle_diff'] = 9 #8
        config_dict['Ipos_sd'] = 5
        config_dict['U_ECstf'] = 0.001

        config_dict['ECstf_rest'] = 0.25
        config_dict['ECstf_target'] = 1.5
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 4
        config_dict['U_stdx_CA3'] = 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0.1
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        return config_dict

    def fig3_NoRecurrence(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 5 #3
        config_dict['Iangle_diff'] = 9 #8
        config_dict['Ipos_sd'] = 5

        config_dict['ECstf_rest'] = 0.25 #0
        config_dict['ECstf_target'] = 1.5 #2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 4000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict

    def fig4_Ctrl_NoRecurrence(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 3
        config_dict['Iangle_diff'] = 8
        config_dict['Ipos_sd'] = 5

        config_dict['ECstf_rest'] = 0
        config_dict['ECstf_target'] = 2
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 4000
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict

    def fig4_DGlesion_NoRecurrence(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 3
        config_dict['Iangle_diff'] = 8
        config_dict['Ipos_sd'] = 5

        config_dict['ECstf_rest'] = 1.25
        config_dict['ECstf_target'] = 1.25
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['U_ECstf'] = 0
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 0
        config_dict['mos_exist'] = False
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict


    def fig5_NoRecurrence(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 6 # 6
        config_dict['Iangle_diff'] = 14 # 8

        config_dict['ECstf_rest'] = 0.25
        config_dict['ECstf_target'] = 1.5
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 5500
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict


    def fig6_NoRecurrence_WithLoop(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 2 # 6
        config_dict['Iangle_diff'] = 14 # 8
        config_dict['Iangle_compen'] = 2
        config_dict['ECstf_rest'] = 0.25
        config_dict['ECstf_target'] = 1.5
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 5500
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict
    def fig6_NoRecurrence_NoLoop(self):
        config_dict = deepcopy(self.config_dict)
        config_dict['Ipos_max'] = 2 # 6
        config_dict['Iangle_diff'] = 14 # 8
        config_dict['Iangle_compen'] = 2
        config_dict['ECstf_rest'] = 0.25
        config_dict['ECstf_target'] = 1.5
        config_dict['nx_mos'] = 40
        config_dict['ny_mos'] = 40
        config_dict['U_stdx_CA3'] = 0.7  # 0.7
        config_dict['wmax_ca3ca3'] = 0
        config_dict['wmax_ca3ca3_adiff'] = 0
        config_dict['w_ca3ca3_akappa'] = 1
        config_dict['asym_flag'] = False
        config_dict['wmax_ca3mosca3_adiff'] = 5500
        config_dict['mos_exist'] = True
        config_dict['wmax_CA3in'] = 50
        config_dict['wmax_inCA3'] = 5
        config_dict['wmax_Mosin'] = 350
        config_dict['wmax_inMos'] = 35
        return config_dict
