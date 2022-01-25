
import pathlib
import copy
import numpy as np
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

from uk.data import GuitarStringData, GuitarBodyData, AFloat, AInt, ACallableFloat
from uk.structure import GuitarString, GuitarBody, Force, ForceRamp, ForceNull, ModalSimulation
from util.util import make_modetime_dataframe, load_data_json, load_data_csv


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

    if log.do_save:
        output_spreadsheet_path.mkdir(parents=True, exist_ok=True)
        output_figure_path.mkdir(parents=True, exist_ok=True)

    if log.do_log:
        print(f"String data: \n {string.data._param_dict}")
        print(f"Body data: \n {body.data._param_dict}")
        print(f"Simulation config \n {sim._param_dict}")

    # SIMULATION
    # There is no external force applied to the body.
    ext_force_body = ForceNull()

    # The string and body are initially at rest.
    q_n_is = [np.zeros(sim.n.shape, dtype=float) for i in range(2)]
    dq_n_is = [np.zeros(sim.n.shape, dtype=float) for i in range(2)]

    # Run the simulation / solve the system.
    t, q_ns, dq_ns, ddq_ns, ext_force_n_ts = sim.run(
        [string, body], [ext_force_string, ext_force_body],
        q_n_is, dq_n_is)

    y_ns = [struct.y_n(q_ns[i], sim.n)
            for (i, struct) in enumerate([string, body])]
    y_n = y_ns[0]

    # compute data frames from the result.
    df_q_n = make_modetime_dataframe(q_ns[0], sim.n, t)
    df_dq_n = make_modetime_dataframe(dq_ns[0], sim.n, t)
    df_ddq_n = make_modetime_dataframe(ddq_ns[0], sim.n, t)
    df_ext_force_n_t = make_modetime_dataframe(
        ext_force_n_ts[0], sim.n, t)

    # save the result as required.
    x = np.linspace(0, string.data.l, log.nb_points)
    xx = np.outer(x, np.ones_like(t))
    tt = np.outer(np.ones_like(x), t)

    if log.do_save:
        df_q_n.to_csv(output_spreadsheet_path / 'q_n.csv')
        df_ddq_n.to_csv(output_spreadsheet_path / 'dq_n.csv')
        df_ddq_n.to_csv(output_spreadsheet_path / 'ddq_n.csv')
        df_ext_force_n_t.to_csv(output_spreadsheet_path / 'ext_force_n_t.csv')

    if log.do_log or log.do_save:
        # EXCITATION FORCE ext_force
        # TODO
        f_x = ext_force_string(xx, tt)
        fig = plt.figure(figsize=(6, 8))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(f'External force')
        ax.plot_surface(xx, tt, f_x)
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$t$ (s)')
        ax.set_zlabel('$F_{ext}(x, t)$ (N)')
        if log.do_log:
            plt.show()
        if log.do_save:
            fig.savefig(output_figure_path / 'ext_force.svg',
                        facecolor='none', transparent=True)
        plt.close(fig)
        # MODAL DISPLACEMENTS y_n
        fig = plt.figure(figsize=(len(y_n) * 3, 8))
        fig.suptitle("Modal displacements (unconstrained)")

        for (j, y_j) in enumerate(y_n):
            ax = fig.add_subplot(2, len(y_n)//2+1, j+1, projection='3d')
            y_x = y_j(xx)
            #
            ax.set_title(f'String mode ${j}$')
            ax.plot_surface(xx, tt, y_x)
            ax.set_xlabel('$x$ (m)')
            ax.set_ylabel('$t$ (s)')
            ax.set_zlabel(f'$y_{j}^S(x, t)$ (m)')
        if log.do_log:
            plt.show()
        if log.do_save:
            fig.savefig(output_figure_path / 'y_n.svg',
                        facecolor='none', transparent=True)
        plt.close(fig)
