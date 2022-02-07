import pathlib
import numpy as np
import numpy.typing as npt
import scipy.io.wavfile as wav
import scipy.signal as sig
from scipy import io

import matplotlib.pyplot as plt
import matplotlib.figure as pltfig

from typing import Tuple, Dict, Union, Any, Callable

from util.util import compute_frf, to_db


def load_analysis(
    path: Union[str, pathlib.Path], conf: object
) -> Dict[str, np.ndarray]:
    """Load analysis

    Args:
        path (Union[str, pathlib.Path]): [description]
        conf (Dict): [description]

    Returns:
        Dict[str, np.ndarray]: [description]
    """
    try:
        data_raw = io.loadmat(path, simplify_cells=True)
    except ValueError:
        return None
    return perform_analysis(data_raw, conf)


def find_win_start(sig: npt.NDArray[float], attack_length: int) -> int:
    """Find the best region of interest for the windowing the signal

    Args:
        sig (npt.NDArray[float]): A temporal signal
        anticipation_length (int): Length to start from the max of the signal

    Returns:
        Tuple[int, int]: [description]
    """
    max_idx = np.argmax(sig)
    idx_start = int(np.floor(max_idx - attack_length))
    return idx_start


def make_linramp(size: int, start: int, stop: int) -> npt.NDArray[float]:
    """Linear force ramp

    Args:
        size (int): [description]
        start (int): [description]
        stop (int): [description]

    Returns:
        npt.NDArray[float]: [description]
    """
    r = np.zeros(size)
    r[start:stop] = np.arange(stop - start) / (stop - start)
    return r


def perform_analysis(
    data_raw: Dict[str, np.ndarray], conf: object
) -> Dict[str, np.ndarray]:
    """Compute stuff again

    Args:
        data (Dict[str, np.ndarray]): [description]

    Returns:
        Dict[np.ndarray]: [description]
    """
    data_raw = data_raw["Final"]
    #
    times = data_raw["time"]
    sr = np.round(1 / (times[1] - times[0])).astype(int)
    n_fft = np.power(2, np.ceil(np.log2(len(times)))).astype(int)
    #
    excitation_kind = "hammer"
    # excitation_kind = "wire"
    # excitation_kind = data_raw["excitation_kind"]
    #
    # Signals and windowed signals
    # -- Microphone
    data_mic = data_raw["micro"]["brut"]
    idx_mic_win_start = find_win_start(data_mic, conf.win.mic.attack_length)
    idx_mic_win_stop = len(times)
    mic_win_kwargs = vars(conf.win.mic.kwargs)
    win_mic = sig.get_window(
        (conf.win.mic.name, *mic_win_kwargs.values()),
        idx_mic_win_stop - idx_mic_win_start,
    )
    times_mic_win = times[idx_mic_win_start:idx_mic_win_stop]
    data_mic_win = win_mic * data_mic[idx_mic_win_start:idx_mic_win_stop]
    # -- Hammer
    idx_excit_win_start = None
    idx_excit_win_stop = None
    data_excit = None
    conf_excit_win = None
    if excitation_kind == "hammer":
        data_excit = data_raw["marteau"]["brut"]
        idx_excit_win_start = find_win_start(data_excit, conf.win.ham.length // 2)
        idx_excit_win_stop = idx_excit_win_start + conf.win.ham.length
        conf_excit_win = conf.win.ham
    elif excitation_kind == "wire":
        # the onset of the signal in the microphone
        # should be the end of the excitation by the wire.
        idx_excit_win_stop = find_win_start(data_mic, conf.win.mic.attack_length)
        idx_excit_win_start = idx_excit_win_stop - conf.win.wire.attack_length
        data_excit = make_linramp(len(times), idx_excit_win_start, idx_excit_win_stop)
        conf_excit_win = conf.win.wire
    kwargs_excit_win = vars(conf_excit_win.kwargs)
    # FIXME: can't choose the args  by keyword in get_window
    win_excit = sig.get_window(
        (conf_excit_win.name, *kwargs_excit_win.values()),
        idx_excit_win_stop - idx_excit_win_start,
    )
    times_excit_win = times[idx_excit_win_start:idx_excit_win_stop]
    data_excit_win = win_excit * data_excit[idx_excit_win_start:idx_excit_win_stop]
    # -- Accelerometer
    data_acc = data_raw["accelero"]["brut"]
    idx_acc_win_start = find_win_start(data_acc, conf.win.acc.attack_length)
    idx_acc_win_stop = len(times)
    acc_win_kwargs = vars(conf.win.acc.kwargs)
    win_acc = sig.get_window(
        (conf.win.acc.name, *acc_win_kwargs.values()),
        idx_acc_win_stop - idx_acc_win_start,
    )
    times_acc_win = times[idx_acc_win_start:idx_acc_win_stop]
    data_acc_win = win_acc * data_acc[idx_acc_win_start:idx_acc_win_stop]
    # Frequential responses
    freqs = np.fft.fftfreq(n_fft) * sr
    freqs_win = np.fft.fftfreq(n_fft) * sr
    # Over a set time frame
    ft_excit = np.fft.fft(data_excit, n=n_fft)
    ft_acc = np.fft.fft(data_acc, n=n_fft)
    ft_mic = np.fft.fft(data_mic, n=n_fft)
    frf = compute_frf(ft_excit, ft_acc)
    imp = np.real(np.fft.ifft(frf, n=n_fft))  # len(times)))
    #
    ft_excit_win = np.fft.fft(data_excit_win, n=n_fft)
    ft_acc_win = np.fft.fft(data_acc_win, n=n_fft)
    ft_mic_win = np.fft.fft(data_mic_win, n=n_fft)
    frf_win = compute_frf(ft_excit_win, ft_acc_win)
    imp_win = np.real(np.fft.ifft(frf_win, n=n_fft))  # len(times)))
    times_imp = np.arange(n_fft) / sr
    # Spectrogrammes
    spec_win = sig.get_window(conf.spec.win.name, conf.spec.win.length)
    freqs_spec, times_spec, spec_excit = sig.spectrogram(
        data_excit,
        nfft=conf.spec.n_fft,
        fs=sr,
        window=spec_win,
        noverlap=conf.spec.n_overlap,
    )
    _, _, spec_acc = sig.spectrogram(
        data_acc,
        nfft=conf.spec.n_fft,
        fs=sr,
        window=spec_win,
        noverlap=conf.spec.n_overlap,
    )
    _, _, spec_mic = sig.spectrogram(
        data_mic,
        nfft=conf.spec.n_fft,
        fs=sr,
        window=spec_win,
        noverlap=conf.spec.n_overlap,
    )

    data = {
        "excitation_kind": excitation_kind,
        "times": {
            "whole": times,
            "excit": {
                "win": times_excit_win,
                "extents": np.array([idx_excit_win_start, idx_excit_win_stop]),
            },
            "acc": {
                "win": times_acc_win,
                "extents": np.array([idx_acc_win_start, idx_acc_win_stop]),
            },
            "mic": {
                "win": times_mic_win,
                "extents": np.array([idx_mic_win_start, idx_mic_win_stop]),
            },
            "imp": times_imp,
            "spec": times_spec,
        },
        "freqs": {"whole": freqs, "win": freqs_win, "spec": freqs_spec},
        "sr": sr,
        "n_fft": n_fft,
    }
    data["temporal"] = {
        "excit": {"whole": data_excit, "win": data_excit_win},
        "acc": {"whole": data_acc, "win": data_acc_win},
        "mic": {"whole": data_mic, "win": data_mic_win},
        "imp": {"whole": imp, "win": imp_win},
    }
    data["frequential"] = {
        "excit": {"whole": ft_excit, "win": ft_excit_win},
        "acc": {"whole": ft_acc, "win": ft_acc_win},
        "mic": {"whole": ft_mic, "win": ft_mic_win},
        "frf": {"whole": frf, "win": frf_win},
    }
    data["spec"] = {"excit": spec_excit, "acc": spec_acc, "mic": spec_mic}
    return data


def compute_analysis_figures(
    data: Dict[str, np.ndarray],
    conf: object,
    fig_fac: Callable[[Any], pltfig.Figure] = plt.figure,
) -> Tuple[pltfig.Figure]:
    """Plots stuff in figures

    Args:
        data (Dict[str, np.ndarray]): [description]
        fig_fac (Callable[[Any], pltfig.Figure], optional): [description]. Defaults to plt.figure.

    Returns:
        Tuple[pltfig.Figure]: [description]
    """
    excitation_kind = data["excitation_kind"]
    #
    times = data["times"]["whole"]
    times_excit_win = data["times"]["excit"]["win"]
    times_acc_win = data["times"]["acc"]["win"]
    times_mic_win = data["times"]["mic"]["win"]
    times_imp = data["times"]["imp"]
    #
    extents_excit_win = data["times"]["excit"]["extents"]
    extents_acc_win = data["times"]["acc"]["extents"]
    extents_mic_win = data["times"]["mic"]["extents"]
    #
    times_spec = data["times"]["spec"]
    freqs = data["freqs"]["whole"]
    freqs_win = data["freqs"]["win"]
    freqs_spec = data["freqs"]["spec"]
    #
    sr = data["sr"]
    #
    sig_excit = data["temporal"]["excit"]["whole"]
    sig_excit_win = data["temporal"]["excit"]["win"]
    sig_acc = data["temporal"]["acc"]["whole"]
    sig_acc_win = data["temporal"]["acc"]["win"]
    sig_mic = data["temporal"]["mic"]["whole"]
    sig_mic_win = data["temporal"]["mic"]["win"]
    imp = data["temporal"]["imp"]["whole"]
    imp_win = data["temporal"]["imp"]["win"]
    #
    ft_excit = data["frequential"]["excit"]["whole"]
    ft_excit_win = data["frequential"]["excit"]["win"]
    ft_acc = data["frequential"]["acc"]["whole"]
    ft_acc_win = data["frequential"]["acc"]["win"]
    ft_mic = data["frequential"]["mic"]["whole"]
    ft_mic_win = data["frequential"]["mic"]["win"]
    frf = data["frequential"]["frf"]["whole"]
    frf_win = data["frequential"]["frf"]["win"]
    #
    spec_excit = data["spec"]["excit"]
    spec_acc = data["spec"]["acc"]
    spec_mic = data["spec"]["mic"]
    #
    freq_focus = (0, 0.5 * sr)
    if conf.focus.freq is not None:
        freq_focus = conf.focus.freq
    time_focus = (times[0], times[-1])
    time_focus_excit = (times[0], times[-1])
    time_focus_acc = (times[0], times[-1])
    time_focus_mic = (times[0], times[-1])
    if conf.focus.time is not None:
        if conf.focus.time.center == "win":
            time_focus_ham = extents_ham_win / sr
            time_focus_acc = extents_acc_win / sr
            time_focus_mic = extents_mic_win / sr
            """
            centre_excit_win = np.mean(extents_excit_win)
            dist_excit_win = np.amax(extents_excit_win) - centre_excit_win
            centre_acc_win = np.mean(extents_acc_win)
            dist_acc_win = np.amax(extents_acc_win) - centre_acc_win
            centre_mic_win = np.mean(extents_mic_win)
            dist_mic_win = np.amax(extents_mic_win) - centre_mic_win
            time_focus_excit = (
                centre_excit_win
                + np.array([-1, 1]) * dist_excit_win * conf.focus.time.ratio
            ) / sr
            time_focus_acc = (
                centre_acc_win
                + np.array([-1, 1]) * dist_acc_win * conf.focus.time.ratio
            ) / sr
            time_focus_mic = (
                centre_mic_win
                + np.array([-1, 1]) * dist_mic_win * conf.focus.time.ratio
            ) / sr
            """
    #  TEMPORAL RESPONSES
    # Excitation
    fig_time = fig_fac(figsize=(16, 12))
    fig_time.subplots_adjust(hspace=0.2, wspace=0.4)
    fig_time.suptitle("Temporal responses")
    #
    ax = fig_time.add_subplot(2, 2, 1)
    ax.set_title(f"Excitation ({excitation_kind})")
    ax.plot(times, sig_excit)
    ax.plot(times_excit_win, sig_excit_win, c="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.set_xlim(time_focus_excit)
    # Accelerometer
    ax = fig_time.add_subplot(2, 2, 2)
    ax.set_title("Accelerometer")
    ax.plot(times, sig_acc)
    ax.plot(times_acc_win, sig_acc_win, c="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m.s$^{-2}$)")
    ax.set_xlim(time_focus_acc)
    # Microphone
    ax = fig_time.add_subplot(2, 2, 3)
    ax.set_title("Microphone")
    ax.plot(times, sig_mic)
    ax.plot(times_mic_win, sig_mic_win, c="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (dB?)")
    ax.set_xlim(time_focus_mic)
    #
    ax = fig_time.add_subplot(2, 2, 4)
    ax.set_title("Impulse response")
    ax.plot(times_imp, imp)
    ax.plot(times_imp, imp_win, c="r")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(time_focus)
    # FREQUENTIAL RESPONSES
    fig_freq = fig_fac(figsize=(16, 12))
    fig_freq.subplots_adjust(hspace=0.3, wspace=0.4)
    fig_freq.suptitle("Frequential responses (modules)")
    #
    ax = fig_freq.add_subplot(3, 1, 1)
    ax.set_title(f"Excitation ({excitation_kind})")
    ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_excit))))
    ax.plot(
        np.fft.fftshift(freqs_win), np.fft.fftshift(to_db(np.abs(ft_excit_win))), c="r"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Force (dB[N])")
    ax.set_xlim(freq_focus)
    # Accelerometer
    ax = fig_freq.add_subplot(3, 1, 2)
    ax.set_title("Accelerometer")
    ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_acc))))
    ax.plot(
        np.fft.fftshift(freqs_win), np.fft.fftshift(to_db(np.abs(ft_acc_win))), c="r"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Acceleration (dB[m.s$^{-2}$]r")
    ax.set_xlim(freq_focus)
    #
    # Microphone
    ax = fig_freq.add_subplot(3, 1, 3)
    ax.set_title("Microphone")
    ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(ft_mic))))
    ax.plot(
        np.fft.fftshift(freqs_win), np.fft.fftshift(to_db(np.abs(ft_mic_win))), c="r"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_xlim(freq_focus)
    #
    # Frequency response functions
    fig_frf = fig_fac(figsize=(16, 12))
    fig_frf.subplots_adjust(hspace=0.3, wspace=0.4)
    fig_frf.suptitle("Frequency response functions")
    # Module
    ax = fig_frf.add_subplot(2, 1, 1)
    ax.set_title("Module")
    ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(to_db(np.abs(frf))))
    ax.plot(np.fft.fftshift(freqs_win), np.fft.fftshift(to_db(np.abs(frf_win))), c="r")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Accelerance (dB)")
    ax.set_xlim(freq_focus)
    # Phase
    # ax = fig_frf.add_subplot(2, 1, 2)
    # ax.set_title("Phase")
    # ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(np.unwrap(np.angle(frf))))
    #
    # ax.plot( np.fft.fftshift(freqs_win), np.fft.fftshift(np.unwrap(np.angle(frf_win))), c="r")
    # ax.set_xlabel("Frequency (Hz)")
    # ax.set_ylabel("Phase (rad)")
    # ax.set_xlim(freq_focus)
    #
    # Spectrogrammes
    spec_extent = [times_spec[0], times_spec[-1], freqs_spec[0], freqs_spec[-1]]
    # Module
    fig_spec = fig_fac(figsize=(16, 12))
    axes = []
    surfs = []
    fig_spec.subplots_adjust(hspace=0.2, wspace=0.4)
    fig_spec.suptitle("Spectrogrammes")
    # Excitation
    ax = fig_spec.add_subplot(3, 1, 1)
    ax.set_title(f"Excitation ({excitation_kind})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    surf = ax.imshow(
        to_db(spec_excit), extent=spec_extent, origin="lower", aspect="auto"
    )
    axes.append(ax)
    surfs.append(surf)
    # Accelerometer
    ax = fig_spec.add_subplot(3, 1, 2)
    ax.set_title("Accelerometer")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    surf = ax.imshow(to_db(spec_acc), extent=spec_extent, origin="lower", aspect="auto")
    axes.append(ax)
    surfs.append(surf)
    # Microphone
    ax = fig_spec.add_subplot(3, 1, 3)
    ax.set_title("Microphone")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    surf = ax.imshow(to_db(spec_mic), extent=spec_extent, origin="lower", aspect="auto")
    axes.append(ax)
    surfs.append(surf)
    cbar = fig_spec.colorbar(surfs[0], ax=axes)
    cbar.ax.set_ylabel("Amplitude (dB)", rotation=270)
    #
    return fig_time, fig_freq, fig_frf, fig_spec


def save_analysis(
    output_path: str, data: Dict[str, np.ndarray], plot_conf: Dict
) -> None:
    """Save analysis output in figures and sounds.

    Args:
        data (Dict[str, np.ndarray]): [description]
        output_path (str): [description]
    """
    fig_time, fig_freq, fig_frf, fig_spec = compute_analysis_figures(
        data, plot_conf, pltfig.Figure
    )
    #
    sr = data["sr"]
    data_mic = data["temporal"]["mic"]["whole"]
    data_acc = data["temporal"]["acc"]["whole"]
    imp = data["temporal"]["imp"]["whole"]
    imp_win = data["temporal"]["imp"]["win"]
    #
    output_path = pathlib.Path(output_path)
    output_audio_path = output_path / "sounds"
    output_figure_path = output_path / "figures"
    output_figure_vec_path = output_figure_path / "vec"
    output_figure_img_path = output_figure_path / "img"
    output_figure_vec_path.mkdir(parents=True, exist_ok=True)
    output_figure_img_path.mkdir(parents=True, exist_ok=True)
    output_audio_path.mkdir(parents=True, exist_ok=True)
    # audio
    wav.write(output_audio_path / "mic.wav", sr, data_mic / np.amax(np.abs(data_mic)))
    wav.write(output_audio_path / "acc.wav", sr, data_acc / np.amax(np.abs(data_acc)))
    wav.write(output_audio_path / "imp.wav", sr, imp / np.amax(np.abs(imp)))
    wav.write(output_audio_path / "imp_win.wav", sr, imp_win / np.amax(np.abs(imp_win)))
    # figures
    fig_time.savefig(
        output_figure_vec_path / "time_responses.svg",
        facecolor="none",
        transparent=True,
    )
    fig_freq.savefig(
        output_figure_vec_path / "freq_responses.svg",
        facecolor="none",
        transparent=True,
    )
    fig_frf.savefig(
        output_figure_vec_path / "frfs.svg", facecolor="none", transparent=True
    )
    fig_spec.savefig(
        output_figure_vec_path / "spectrogrammes.svg",
        facecolor="none",
        transparent=True,
    )
    #
    fig_time.savefig(output_figure_img_path / "time_responses.png")
    fig_freq.savefig(output_figure_img_path / "freq_responses.png")
    fig_frf.savefig(output_figure_img_path / "frfs.png")
    fig_spec.savefig(output_figure_img_path / "spectrogrammes.png")
