
import pathlib
import copy
import numpy as np
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
import json
import argparse

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
    parser.add_argument('--out_dir', default=None, type=str,
                        help='Output directory for output data')
    args = parser.parse_args()

    # Loading config / data
    string_data = load_data_json(GuitarStringData, args.string)
    body_data = load_data_csv(GuitarBodyData, args.body)
    ext_force_string = load_data_json(
        ForceRamp, args.excitation, l=string_data.l)
    sim = load_data_json(ModalSimulation, args.simulation)

    # SIMULATION
    body = GuitarBody(body_data)
    string = GuitarString(string_data)

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

    # compute data frames from the result.
    df_q_n = make_modetime_dataframe(q_ns[0], sim.n, t)
    df_dq_n = make_modetime_dataframe(dq_ns[0], sim.n, t)
    df_ddq_n = make_modetime_dataframe(ddq_ns[0], sim.n, t)
    df_ext_force_n_t = make_modetime_dataframe(
        ext_force_n_ts[0], sim.n, t)

    # save the result as required.
    if args.out_dir is not None:
        output_path = pathlib.Path(args.out_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        df_q_n.to_csv(output_path / 'q_n.csv')
        df_ddq_n.to_csv(output_path / 'dq_n.csv')
        df_ddq_n.to_csv(output_path / 'ddq_n.csv')
        df_ext_force_n_t.to_csv(output_path / 'ext_force_n_t.csv')
