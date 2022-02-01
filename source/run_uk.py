
import pathlib
import copy
import numpy as np
import scipy.io.wavfile as wav
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
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
    parser.add_argument('--finger_pos', default=0, type=float,
                        help='Finger position on soundboard')
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

    # SIMULATION
    # There is no external force applied to the body.
    ext_force_body = ForceNull()
    # List of finger constraints, there is none on the body
    finger_constraints = [args.finger_pos, 0]

    # The string and body are initially at rest.
    q_n_is = [np.zeros(sim.n[i].shape, dtype=float) for i in range(2)]
    dq_n_is = [np.zeros(sim.n[i].shape, dtype=float) for i in range(2)]

    # Run the simulation / solve the system.
    t, q_ns, dq_ns, ddq_ns, ext_force_n_ts = sim.run(
        [string, body], [ext_force_string, ext_force_body],
        q_n_is, dq_n_is, finger_constraints)

    y_ns = [struct.y_n(q_ns[i], sim.n[i])
            for (i, struct) in enumerate([string, body])]

    # compute data frames from the result.
    df_q_n = pd.DataFrame(q_ns[0], index=sim.n[0], columns=t)
    df_dq_n = pd.DataFrame(dq_ns[0], index=sim.n[0], columns=t)
    df_ddq_n = pd.DataFrame(ddq_ns[0], index=sim.n[0], columns=t)
    df_ext_force_n_t = pd.DataFrame(
        ext_force_n_ts[0], index=sim.n[0], columns=t)

    # save the result as required.
    x = np.linspace(0, string.data.l, log.plot.nb_points)
    xx = np.outer(x, np.ones_like(t))
    tt = np.outer(np.ones_like(x), t)

    y_n = y_ns[0]
    ext_force_n_t = ext_force_n_ts[0]

    # Get the total displacement from the sum of the modal displacements.
    y = np.zeros_like(t)
    for j in range(len(y_n)):
        y += y_n[j](log.audio.x_s_rel * string.data.l)

    if log.do_save:
        df_q_n.to_csv(output_spreadsheet_path / 'q_n.csv')
        df_ddq_n.to_csv(output_spreadsheet_path / 'dq_n.csv')
        df_ddq_n.to_csv(output_spreadsheet_path / 'ddq_n.csv')
        df_ext_force_n_t.to_csv(output_spreadsheet_path / 'ext_force_n_t.csv')

    if log.do_log or log.do_save:
        # EXCITATION FORCE ext_force
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        f_x = ext_force_string(xx, tt)
        surf = ax.plot_surface(xx, tt, f_x, cmap='coolwarm')
        ax.set_title(f'Excitation force applied to the string')
        ax.set_xlabel('$x$ (m)')
        ax.set_ylabel('$t$ (s)')
        ax.set_zlabel('$F_{ext}(x, t)$ (N)')
        fig.colorbar(surf, ax=ax)
        if log.do_save:
            fig.savefig(output_figure_path / 'ext_force.svg',
                        facecolor='none', transparent=True)
        if log.do_log:
            plt.show()
        plt.close(fig)

        # TOTAL DISPLACEMENT of the string
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        surf = ax.plot(t, y)
        ax.set_title(f"Total displacement of the string at {log.audio.x_s_rel}")
        ax.set_xlabel('$t$ (s)')
        ax.set_ylabel('$y(t)$ (m)')
        if log.do_save:
            fig.savefig(output_figure_path / 'tot_disp.svg',
                        facecolor='none', transparent=True)
        if log.do_log:
            plt.show()
        plt.close(fig)

        # MODAL DISPLACEMENTS of the String y_n
        if log.modal_plots:
            fig = plt.figure(figsize=(8 * (len(y_n)+1)//2, 2*6))
            fig.subplots_adjust(hspace=0.1, wspace=0.4)
            fig.suptitle("Modal displacements of the string (unconstrained)")
            axes = []
            surfs = []
            for (j, y_j) in enumerate(y_n):
                ax = fig.add_subplot(2, len(y_n)//2+1, j+1, projection='3d')
                axes.append(ax)
                y_x = y_j(xx)
                #
                surf = ax.plot_surface(xx, tt, y_x, cmap='coolwarm')
                surfs.append(surf)
                #
                ax.set_title(f'$n={j}$')
                ax.set_xlabel('$x$ (m)')
                ax.set_ylabel('$t$ (s)')
                ax.set_zlabel(f'$y_{j}^S(x, t)$ (m)')
            # add heat map
            fig.colorbar(surfs[0], ax=axes)
            if log.do_save:
                fig.savefig(output_figure_path / 'y_n.svg',
                            facecolor='none', transparent=True)
                wav.write(output_audio_path / 'y.wav', int(1/sim.h), y)
            if log.do_log:
                plt.show()

            plt.close(fig)

        # MODAL Excitation ext_force_n_t
        if log.modal_plots:
            fig = plt.figure(figsize=(8 * (ext_force_n_t.shape[0]+1)//2, 2*6))
            fig.subplots_adjust(hspace=0.3, wspace=0.4)
            fig.suptitle("Modal excitation force to the string")
            for (j, ext_force_j) in enumerate(ext_force_n_t):
                ax = fig.add_subplot(2, len(ext_force_n_t) //
                                     2+1, j+1)
                #
                ax.plot(t, ext_force_j)
                #
                ax.set_title(f'$n={j}$')
                ax.set_xlabel('$t$ (s)')
                ax.set_ylabel(f'$F_{j}^S(t)$ (N)')
            if log.do_save:
                fig.savefig(output_figure_path / 'ext_force_n.svg',
                            facecolor='none', transparent=True)
            if log.do_log:
                plt.show()
            plt.close(fig)
