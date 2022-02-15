from tqdm import tqdm
import pathlib
import copy
import numpy as np
import scipy.io.wavfile as wav
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
import pickle
import json
import argparse
import matplotlib.pyplot as plt

from uk.data import GuitarStringData, GuitarBodyData, AFloat, AInt, ACallableFloat
from uk.structure import GuitarString, GuitarBody, Force, ForceRamp, ForceNull, ModalSimulation
from util.util import load_data_json, load_data_csv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Guitar test')
    parser.add_argument('--string', required=True, type=str,
                        help='Guitar string config file path')
    parser.add_argument('--body', required=True, type=str,
                        help='Guitar body data file path')
    parser.add_argument('--excitation', required=True, type=str,
                        help='String excitation config file path')
    parser.add_argument('--simulation', required=True, type=str,
                        help='Simulation config file path')
    parser.add_argument('--log', required=True, type=str,
                        help='Logging config file path')
    parser.add_argument('--out_dir', default=None, type=str,
                        help='Output directory for output data')
    # parser.add_argument('--finger_pos', default=0, type=float,
    #                    help='Finger position on soundboard')
    args = parser.parse_args()

    # Loading config from config files
    string = GuitarString(load_data_json(args.string, cls=GuitarStringData))
    body = GuitarBody(load_data_csv(args.body, cls=GuitarBodyData))
    ext_force_string = load_data_json(
        args.excitation, cls=ForceRamp, l=string.data.l)
    sim = load_data_json(args.simulation, cls=ModalSimulation)
    log = load_data_json(args.log)

    output_path = pathlib.Path(args.out_dir)
    output_spreadsheet_path = output_path / 'spreadsheets'
    output_figure_path = output_path / 'figures'
    output_audio_path = output_path / 'sounds'

    if log.do_save:
        output_spreadsheet_path.mkdir(parents=True, exist_ok=True)
        output_figure_path.mkdir(parents=True, exist_ok=True)
        output_audio_path.mkdir(parents=True, exist_ok=True)

    if log.do_log:
        print(f"String data: \n {string.data._param_dict}")
        print(f"Body data: \n {body.data._param_dict}")
        print(f"Simulation config \n {sim._param_dict}")

    fmax_S = string.f_n(sim.n[0][-1])
    fmax_B = body.f_n(sim.n[1][-1])
    if 2*np.pi*fmax_S*sim.h > 2 or 2*np.pi*fmax_B*sim.h > 2:
        raise ValueError("Simulation is going to be unstable, reduce the time step.")

    # SIMULATION
    # There is no external force applied to the body.
    ext_force_body = np.array([lambda x: 0]*len(sim.n[1]), dtype=type(Callable))

    # create modal matrices
    c_n = np.hstack((string.c_n(sim.n[0]), body.c_n(sim.n[1])))
    k_n = np.hstack((string.k_n(sim.n[0]), body.k_n(sim.n[1])))
    m_n = np.hstack((string.m_n(sim.n[0]), body.m_n(sim.n[1])))
    m_halfinv_mat = np.diag(np.power(m_n, -0.5))
    c_n = c_n[:, np.newaxis]
    m_n = m_n[:, np.newaxis]
    k_n = k_n[:, np.newaxis]

    a_ns = []
    a_bridge = string.bridge_coupling(sim.n[0], sim.coupling)
    a_finger = string.finger_coupling(sim.n[0], sim.finger)
    a_ns.append(np.vstack((a_bridge, a_finger)))
    a_bridge = body.bridge_coupling(sim.n[1], sim.coupling)
    a_finger = body.finger_coupling(sim.n[1], 0)
    a_ns.append(np.vstack((a_bridge, a_finger)))
    a_mat = np.hstack(a_ns)
    b_mat = a_mat @ m_halfinv_mat
    b_plus_mat = b_mat.T@(np.linalg.inv(b_mat@b_mat.T))
    w_mat = sim.solve_constraints([string, body], a_mat,
                                  m_halfinv_mat, b_plus_mat)
    # Init vectors
    num_modes = [len(modes) for modes in sim.n]
    q_n = np.zeros((np.sum(num_modes), 1), dtype=float)
    dq_n = np.zeros((np.sum(num_modes), 1), dtype=float)
    ddq_n = np.zeros((np.sum(num_modes), 1), dtype=float)
    force_n = np.zeros((np.sum(num_modes), 1), dtype=float)

    # Initialize files
    file_q_n_path = output_spreadsheet_path / 'q_n.pkl'
    file_dq_n_path = output_spreadsheet_path / 'dq_n.pkl'
    file_ddq_n_path = output_spreadsheet_path / 'ddq_n.pkl'
    file_force_n_t_path = output_spreadsheet_path / 'force_n_t.pkl'

    f_q_n = open(file_q_n_path, 'wb')
    f_dq_n = open(file_dq_n_path, 'wb')
    f_ddq_n = open(file_ddq_n_path, 'wb')
    f_force_n = open(file_force_n_t_path, 'wb')
    # FIRST_LINE = 't,'+','.join([f"{n}" for n in sim.n[0]]) + '\n'
    # f_q_n.write(FIRST_LINE)
    # f_dq_n.write(FIRST_LINE)
    # f_ddq_n.write(FIRST_LINE)
    # f_force_n.write(FIRST_LINE)
    # f_q_n.write('0,')
    # q_n[:num_modes[0]].tofile(f_q_n, sep=",")
    # f_dq_n.write('0,')
    # f_ddq_n.write('0,')
    # f_force_n.write('0,')
    # dq_n[:num_modes[0]].tofile(f_dq_n, sep=",")
    # ddq_n[:num_modes[0]].tofile(f_ddq_n, sep=",")
    # force_n[:num_modes[0]].tofile(f_force_n, sep=",")
    pickle.dump((0, q_n), f_q_n)
    pickle.dump((0, dq_n), f_dq_n)
    pickle.dump((0, ddq_n), f_ddq_n)
    pickle.dump((0, force_n), f_force_n)
    # Run the simulation / solve the system.
    ext_force_n = string.ext_force_n(ext_force_string, sim.n[0])
    ext_force_n = np.hstack((ext_force_n, ext_force_body))
    for s in tqdm(range(1, sim.nb_steps), desc="Computing..."):
        t = s*sim.h
        for (j, force) in enumerate(ext_force_n):
            force_n[j] = force(t)
        q_n, dq_n, ddq_n, force_n = sim.step(q_n, dq_n, ddq_n, force_n,
                                             c_n, k_n, m_n, m_halfinv_mat,
                                             a_mat, b_plus_mat, w_mat)
        pickle.dump((t, q_n), f_q_n)
        pickle.dump((t, dq_n), f_dq_n)
        pickle.dump((t, ddq_n), f_ddq_n)
        pickle.dump((t, force_n), f_force_n)
        # f_q_n.write(f"\n{t},")
        # q_n[:num_modes[0]].tofile(f_q_n, sep=',')
        # f_dq_n.write(f"\n{t},")
        # dq_n[:num_modes[0]].tofile(f_dq_n, sep=',')
        # f_ddq_n.write(f"\n{t},")
        # ddq_n[:num_modes[0]].tofile(f_ddq_n, sep=',')
        # f_force_n.write(f"\n{t},")
        # force_n[:num_modes[0]].tofile(f_force_n, sep=',')

    # close files
    f_q_n.close()
    f_dq_n.close()
    f_ddq_n.close()
    f_force_n.close()

    # save config files
    with open(output_path / 'string.pkl', 'wb') as f_string:
        pickle.dump(f_string, string)
    with open(output_path / 'body.pkl', 'wb') as f_body:
        pickle.dump(f_body, body)
    with open(output_path / 'excitation.pkl', 'wb') as f_excitation:
        pickle.dump(f_excitation, ext_force_string)
    with open(output_path / 'sim.pkl', 'wb') as f_sim:
        pickle.dump(f_sim, sim)


