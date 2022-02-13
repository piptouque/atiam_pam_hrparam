import os
import pathlib
from tqdm import tqdm

import argparse

from expr.analysis import load_analysis, save_analysis, compute_analysis_figures
from util.util import load_data_json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experimental data test")
    parser.add_argument(
        "--data_dir", required=True, type=str, help="Experimental data dir path"
    )
    parser.add_argument(
        "--log", required=True, type=str, help="Logging config file path"
    )
    parser.add_argument(
        "--config", required=True, type=str, help="Analysis config file path"
    )
    parser.add_argument(
        "--out_dir", default=None, type=str, help="Output directory for output data"
    )
    args = parser.parse_args()

    # Loading config from config files
    log = load_data_json(args.log)
    conf = load_data_json(args.config)

    # Loading experimental data
    # traverse root directory, and list directories as dirs and files as files
    # see: https://stackoverflow.com/a/16974952
    data_path = pathlib.Path(args.data_dir)
    # count number of files first
    n_files = sum(len(files) for _, _, files in os.walk(data_path))
    if log.do_log or log.do_save:
        # show progress with os.walk.
        # see: https://stackoverflow.com/a/2165062
        with tqdm(os.walk(data_path), total=n_files) as it:
            # progress_list = {}
            for root, dirs, files in it:
                # path = root.split(os.sep)
                root = pathlib.Path(root)
                root_out = pathlib.Path(args.out_dir) / os.path.relpath(root, data_path)
                # progress = calc_progress(progress_list, root, dirs)
                for file in files:
                    data = load_analysis(root / file, conf)
                    if data is None:
                        continue
                    file_name, _ = os.path.splitext(file)
                    if log.do_save:
                        save_analysis(root_out / file_name, data, log.plot)
                    if log.do_log:
                        (
                            fig_time,
                            fig_freq,
                            fig_frf,
                            fig_spec,
                        ) = compute_analysis_figures(data, log.plot)
                    it.update()
