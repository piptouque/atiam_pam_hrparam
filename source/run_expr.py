
import pathlib
import copy
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
from scipy import io
from itertools import chain

from typing import Callable, Union, Tuple, List
from abc import abstractmethod, abstractproperty

import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

from expr.util import find_win_extends
from util.util import load_data_json, compute_frf, to_db


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experimental data test')
    parser.add_argument('--data', required=True, type=str,
                        help='Experimental data file path')
    parser.add_argument('--log', required=True, type=str,
                        help='Logging config file path')
    parser.add_argument('--out_dir', default=None, type=str,
                        help='Output directory for output data')
    args = parser.parse_args()

    # Setting output paths
    output_path = pathlib.Path(args.out_dir)
    output_spreadsheet_path = output_path / 'spreadsheets'
    output_figure_path = output_path / 'figures'
    output_audio_path = output_path / 'sounds'

    # Loading config from config files
    log = load_data_json(args.log)

    # Loading experimental data
    data = io.loadmat(args.data, simplify_cells=True)
    data = data['Final']
    times = data['time']
    sr = np.round(1 / (times[1] - times[0])).astype(int)
    #
    data_ham = data['marteau']['brut']
    data_acc = data['accelero']['brut']
    data_mic = data['micro']['brut']
    #
    win_length = 80
    idx_win_start, idx_win_stop = find_win_extends(data_ham, win_length)
    times_win = times[idx_win_start:idx_win_stop]
    #
    # data_ham_win = data['marteau']['fen']
    # data_acc_win = data['accelero']['fen']
    # data_mic_win = data['micro']['fen']
    #
    data_ham_win = data_ham[idx_win_start:idx_win_stop]
    data_acc_win = data_acc[idx_win_start:idx_win_stop]
    data_mic_win = data_mic[idx_win_start:idx_win_stop]
    #
    n_fft = np.power(2, np.ceil(np.log2(len(data_acc)))).astype(int)
    freqs = np.fft.fftfreq(n_fft) * sr
    n_fft_win = np.power(2, np.ceil(np.log2(len(data_acc_win)))).astype(int)
    freqs_win = np.fft.fftfreq(n_fft_win) * sr
    # Over a set time frame
    ft_ham = np.fft.fft(data_ham, n=n_fft)
    ft_acc = np.fft.fft(data_acc, n=n_fft)
    ft_mic = np.fft.fft(data_mic, n=n_fft)
    frf_res = compute_frf(ft_ham, ft_acc)
    # frf_res = data['FRF']
    # frf_res = np.concatenate((data['FRF'], np.flip(data['FRF'])))
    #
    ft_ham_win = np.fft.fft(data_ham_win, n=n_fft_win)
    ft_acc_win = np.fft.fft(data_acc_win, n=n_fft_win)
    ft_mic_win = np.fft.fft(data_mic_win, n=n_fft_win)
    frf_res_win = compute_frf(ft_ham_win, ft_acc_win)
    # Spectrogrammes
    spec_win = sig.get_window(log.spec.win, log.spec.win_length)
    spec_freqs, spec_times, spec_ham = sig.spectrogram(
        data_ham, nfft=log.spec.n_fft, fs=sr, window=spec_win, noverlap=log.spec.n_overlap)
    _, _, spec_acc = sig.spectrogram(
        data_acc, nfft=log.spec.n_fft, fs=sr, window=spec_win, noverlap=log.spec.n_overlap)
    _, _, spec_mic = sig.spectrogram(
        data_mic, nfft=log.spec.n_fft, fs=sr, window=spec_win, noverlap=log.spec.n_overlap)

    if log.do_save:
        output_spreadsheet_path.mkdir(parents=True, exist_ok=True)
        output_figure_path.mkdir(parents=True, exist_ok=True)
        output_audio_path.mkdir(parents=True, exist_ok=True)

    if log.do_log or log.do_save:
        #  TEMPORAL RESPONSES
        # Hammer
        fig = plt.figure(figsize=(16, 12))
        fig.subplots_adjust(hspace=0.1, wspace=0.4)
        fig.suptitle("Temporal responses")
        #
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title('Hammer')
        ax.plot(times, data_ham)
        ax.plot(times_win, data_ham_win, c='r')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')
        # Accelerometer
        ax = fig.add_subplot(3, 1, 2)
        ax.set_title('Accelerometer')
        ax.plot(times, data_acc)
        ax.plot(times_win, data_acc_win, c='r')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (m.s$^{-2}$)')
        # Microphone
        ax = fig.add_subplot(3, 1, 3)
        ax.set_title('Microphone')
        ax.plot(times, data_mic)
        ax.plot(times_win, data_mic_win, c='r')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (dB?)')
        #
        if log.do_log:
            plt.show()
        plt.close(fig)
        # FREQUENTIAL RESPONSES
        fig = plt.figure(figsize=(16, 12))
        # fig = plt.figure(figsize=(8 * (len(y_n)+1)//2, 2*6))
        fig.subplots_adjust(hspace=0.1, wspace=0.4)
        fig.suptitle("Frequential responses (modules)")
        #
        ax = fig.add_subplot(3, 1, 1)
        ax.set_title('Hammer')
        ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_ham))))
        ax.plot(np.fft.fftshift(freqs_win),
                np.fft.fftshift(to_db(np.abs(ft_ham_win))), c='r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Force (dB[N])')
        ax.set_xlim([0, 0.5 * sr])
        # if log.do_save:
        #   fig.savefig(output_figure_path / 'ext_force.svg', facecolor = 'none', transparent = True)
        # Accelerometer
        ax = fig.add_subplot(3, 1, 2)
        ax.set_title('Accelerometer')
        ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_acc))))
        ax.plot(np.fft.fftshift(freqs_win),
                np.fft.fftshift(to_db(np.abs(ft_acc_win))), c='r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Acceleration (dB[m.s$^{-2}$]r')
        ax.set_xlim([0, 0.5 * sr])
        #
        # Microphone
        ax = fig.add_subplot(3, 1, 3)
        ax.set_title('Microphone')
        ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_mic))))
        ax.plot(np.fft.fftshift(freqs_win),
                np.fft.fftshift(to_db(np.abs(ft_mic_win))), c='r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_xlim([0, 0.5 * sr])
        #
        if log.do_log:
            plt.show()
        plt.close(fig)
        ##
        # Frequency response functions
        fig = plt.figure(figsize=(16, 12))
        # fig = plt.figure(figsize=(8 * (len(y_n)+1)//2, 2*6))
        fig.subplots_adjust(hspace=0.1, wspace=0.4)
        fig.suptitle("Frequency response functions")
        # Module
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title('Module')
        ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(frf_res))))
        ax.plot(np.fft.fftshift(freqs_win),
                np.fft.fftshift(to_db(np.abs(frf_res_win))), c='r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Accelerance (dB)')
        ax.set_xlim([0, 0.5 * sr])
        # Argument
        ax = fig.add_subplot(2, 1, 2)
        ax.set_title('Phase')
        ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.angle(frf_res)))
        ax.plot(np.fft.fftshift(freqs_win), np.fft.fftshift(
            np.angle(frf_res_win)), c='r')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase (rad)')
        ax.set_xlim([0, 0.5 * sr])
        #
        if log.do_log:
            plt.show()
        plt.close(fig)
        # Spectrogrammes
        ff, tt = np.meshgrid(spec_times, spec_freqs)
        spec_extent = [spec_times[0], spec_times[-1],
                       spec_freqs[-1], spec_freqs[0]]
        # Module
        fig = plt.figure(figsize=(16, 12))
        axes = []
        surfs = []
        # fig = plt.figure(figsize=(8 * (len(y_n)+1)//2, 2*6))
        fig.subplots_adjust(hspace=0.1, wspace=0.4)
        fig.suptitle("Spectrogrammes")
        # Hammer
        ax = fig.add_subplot(3, 1, 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        surf = ax.imshow(to_db(spec_ham), extent=spec_extent,
                         origin='lower', aspect='auto')
        axes.append(ax)
        surfs.append(surf)
        # ax.set_zlabel('Amplitude (...)')
        ax = fig.add_subplot(3, 1, 2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (dB[m.s$^{-2}$])')
        surf = ax.imshow(to_db(spec_acc), extent=spec_extent,
                         origin='lower', aspect='auto')
        axes.append(ax)
        surfs.append(surf)
        #
        ax = fig.add_subplot(3, 1, 3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (dB)')
        surf = ax.imshow(to_db(spec_mic), extent=spec_extent,
                         origin='lower', aspect='auto')
        axes.append(ax)
        surfs.append(surf)
        fig.colorbar(surfs[0], ax=axes)
        #
        if log.do_log:
            plt.show()
        plt.close(fig)
