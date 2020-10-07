# DiffWave

 modified by Hyungon Ryu

DiffWave is a fast, high-quality neural vocoder and waveform synthesizer. It starts with Gaussian noise and converts it into speech via iterative refinement. The speech can be controlled by providing a conditioning signal (e.g. log-scaled Mel spectrogram). The model and architecture details are described in [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf).


### Inference API
Basic usage:

```python
from diffwave.inference import predict as diffwave_predict

model_dir = '/path/to/model/dir'
spectrogram = # get your hands on a spectrogram in [N,C,W] format
audio, sample_rate = diffwave_predict(spectrogram, model_dir)

# audio is a GPU tensor in [N,T] format.
```

### Inference CLI
```
python -m diffwave.inference /path/to/model /path/to/spectrogram -o output.wav
```


### Training
Before you start training, you'll need to prepare a training dataset. The dataset can have any directory structure as long as the contained .wav files are 16-bit mono (e.g. [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [VCTK](https://pytorch.org/audio/_modules/torchaudio/datasets/vctk.html)). By default, this implementation assumes a sample rate of 22.05 kHz. If you need to change this value, edit [params.py](https://github.com/lmnt-com/diffwave/blob/master/src/diffwave/params.py).

```
python preprocess.py /path/to/dir/containing/wavs
python train.py /path/to/model/dir /path/to/dir/containing/wavs

# in another shell to monitor training progress:
tensorboard --logdir /path/to/model/dir --bind_all
```

You should expect to hear intelligible (but noisy) speech by ~8k steps (~1.5h on a 2080 Ti).

#### Multi-GPU training
By default, this implementation uses as many GPUs in parallel as returned by [`torch.cuda.device_count()`](https://pytorch.org/docs/stable/cuda.html#torch.cuda.device_count). You can specify which GPUs to use by setting the [`CUDA_DEVICES_AVAILABLE`](https://developer.nvidia.com/blog/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable before running the training module.


## References
- [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/pdf/2009.09761.pdf)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Code for Denoising Diffusion Probabilistic Models](https://github.com/hojonathanho/diffusion)
