import os
import glob
import sys
import logging
import subprocess
import numpy as np
import torch
import torchaudio

from clearml import Task
from utils.hparams import HParams

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    del checkpoint_dict
    torch.cuda.empty_cache()
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({"model": state_dict, "iteration": iteration, "optimizer": optimizer.state_dict(), "learning_rate": learning_rate}, checkpoint_path)


def init_clearml_task(project_name: str, task_name: str, hps: HParams) -> Task:
    """Initializes a new ClearML task and connects hyperparameters."""
    task = Task.init(project_name=project_name, task_name=task_name, auto_connect_frameworks=False)
    task.connect(hps, name="Hyperparameters")
    return task


def log_metrics_to_clearml(clearml_logger, global_step, losses, include_gpu_metrics=False):
    """Logs a dictionary of metrics to the ClearML task."""
    for title, series_data in losses.items():
        for series, value in series_data.items():
            clearml_logger.report_scalar(title=title, series=series, value=value, iteration=global_step)

    # Optionally log GPU metrics
    if include_gpu_metrics:
        gpu_metrics = get_gpu_metrics()
        if gpu_metrics:
            for metric_name, value in gpu_metrics.items():
                # Group GPU metrics by type
                if "Utilization" in metric_name and "Memory" not in metric_name:
                    clearml_logger.report_scalar(title="GPU Utilization", series=metric_name, value=value, iteration=global_step)
                elif "Memory" in metric_name:
                    clearml_logger.report_scalar(title="GPU Memory", series=metric_name, value=value, iteration=global_step)
                elif "Temperature" in metric_name:
                    clearml_logger.report_scalar(title="GPU Temperature", series=metric_name, value=value, iteration=global_step)


def upload_checkpoint_to_clearml(checkpoint_path: str, checkpoint_name: str):
    """Uploads a model checkpoint as an artifact to the current ClearML task."""
    task = Task.current_task()
    if task:
        task.upload_artifact(name=checkpoint_name, artifact_object=checkpoint_path)
    else:
        logger.warning("No active ClearML task found. Skipping artifact upload.")


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, sample_rate=22050):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, sample_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_vocab(vocab_file: str):
    """Load vocabulary from text file
    Args:
        vocab_file (str): Path to vocabulary file
    Returns:
        torchtext.vocab.Vocab: Vocabulary object
    """
    from torchtext.vocab import vocab as transform_vocab
    from text.symbols import UNK_ID, special_symbols

    vocab = {}
    with open(vocab_file, "r") as f:
        for line in f:
            token, index = line.split()
            vocab[token] = int(index)
    vocab = transform_vocab(vocab, specials=special_symbols)
    vocab.set_default_index(UNK_ID)
    return vocab


def save_vocab(vocab, vocab_file: str):
    """Save vocabulary as token index pairs in a text file, sorted by the indices
    Args:
        vocab (torchtext.vocab.Vocab): Vocabulary object
        vocab_file (str): Path to vocabulary file
    """
    with open(vocab_file, "w") as f:
        for token, index in sorted(vocab.get_stoi().items(), key=lambda kv: kv[1]):
            f.write(f"{token}\t{index}\n")


def load_wav_to_torch(full_path):
    """Load wav file
    Args:
        full_path (str): Full path of the wav file

    Returns:
        waveform (torch.FloatTensor): Stereo audio signal [channel, time] in range [-1, 1]
        sample_rate (int): Sampling rate of audio signal (Hz)
    """
    waveform, sample_rate = torchaudio.load(full_path)
    return waveform, sample_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(source_dir))
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn("git hash values are different. {}(saved) != {}(current)".format(saved_hash[:8], cur_hash[:8]))
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def get_gpu_metrics():
    """Captures GPU metrics using nvidia-ml-py if available."""
    try:
        import pynvml
        pynvml.nvmlInit()

        gpu_metrics = {}
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # GPU Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_metrics[f"GPU_{i}_Utilization"] = utilization.gpu
            gpu_metrics[f"GPU_{i}_Memory_Utilization"] = utilization.memory

            # Memory Usage
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_metrics[f"GPU_{i}_Memory_Used_MB"] = memory_info.used / 1024**2
            gpu_metrics[f"GPU_{i}_Memory_Total_MB"] = memory_info.total / 1024**2
            gpu_metrics[f"GPU_{i}_Memory_Free_MB"] = memory_info.free / 1024**2
            gpu_metrics[f"GPU_{i}_Memory_Usage_Percent"] = (memory_info.used / memory_info.total) * 100

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_metrics[f"GPU_{i}_Temperature_C"] = temp
            except:
                pass

        return gpu_metrics

    except ImportError:
        logging.warning("pynvml not available. Install with: pip install nvidia-ml-py")
        return {}
    except Exception as e:
        logging.warning(f"Failed to get GPU metrics: {e}")
        return {}
