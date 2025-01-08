# Pandas Audio Methods

Audio methods for pandas dataframes using soundfile.

Features:

* Use `sf.SoundFile` objects in pandas dataframes
* Call `sf.SoundFile` methods on a column, for example:
  * `.read()`
  * `.truncate()`
  * `.seek()`
* Save dataframes with `sf.SoundFile` objects to Parquet
* Process Audios in parallel with Dask
* Manipulate Audio datasets from Hugging Face

## Installation

```pip
pip install pandas-audio-methods
```

## Usage

You can open Audios as `sf.SoundFile` objects using the `.open()` method.

Once the Audios are opened, you can call any [sf.SoundFile method](https://pillow.readthedocs.io/en/stable/reference/Audio.html#the-Audio-class):

```python
import pandas as pd
from pandas_audio_methods import SFMethods

pd.api.extensions.register_series_accessor("sf")(SFMethods)

df = pd.DataFrame({"file_path": ["path/to/audio.wav"]})
df["audio"] = df["file_path"].sf.open()
# 0    SoundFile('path/to/audio.wav', mode='r', sampl...
# Name: audio, dtype: object, soundfile methods enabled
```

Use with `librosa`:

```python
import librosa
df["audio"] = [librosa.load(audio, sr=16_000) for audio in df["audio"]]
df["audio"] = df["audio"].sf.write()
# 0    SoundFile(<_io.BytesIO object at 0x11b747ba0>,...
# Name: audio, dtype: object, soundfile methods enabled
```

Here is how to enable `sf` methods for `sf.SoundFiles` created manually:

```python
df = pd.DataFrame({"audio": [sf.SoundFile.open("path/to/audio.wav")]})
df["audio"] = df["audio"].sf.enable()
# 0    SoundFile('path/to/audio.wav', mode='r', sampl...
# Name: Audio, dtype: object, soundfile methods enabled
```

## Save

You can save a dataset of `sf.SoundFiles` to Parquet:

```python
# Save
df = pd.DataFrame({"file_path": ["path/to/audio.wav"]})
df["audio"] = df["file_path"].sf.open()
df.to_parquet("data.parquet")

# Later
df = pd.read_parquet("data.parquet")
df["audio"] = df["audio"].sf.enable()
```

This doesn't just save the paths to the Audio files, but the actual Audios themselves !

Under the hood it saves dictionaries of `{"bytes": <bytes of the Audio file>, "path": <path or name of the Audio file>}`.
The Audios are saved as bytes using their Audio encoding or PNG by default. Anyone can load the Parquet data even without `pandas-audio-methods` since it doesn't rely on extension types.

Note: if you created the `sf.SoundFiles` manually, don't forget to enable the `sf` methods to enable saving to Parquet.

## Run in parallel

Dask DataFrame parallelizes pandas to handle large datasets. It enables faster local processing with multiprocessing as well as distributed large scale processing. Dask mimics the pandas API:

```python
import dask.dataframe as dd
from distributed import Client
from pandas_audio_methods import SFMethods

dd.extensions.register_series_accessor("sf")(SFMethods)

if __name__ == "__main__":
    client = Client()
    df = dd.read_csv("path/to/large/dataset.csv")
    df = df.repartition(npartitions=1000)  # divide the processing in 1000 jobs
    df["audio"] = df["file_path"].sf.open()
    df["audio"].head(1)
    # 0    SoundFile('path/to/audio.wav', mode='r', sampl...
    # Name: audio, dtype: object, soundfile methods enabled
    df.to_parquet("data_folder")
```

## Hugging Face support

Most Audio datasets in Parquet format on Hugging Face are compatible with `pandas-audio-methods`. For example you can load the microset of the [People's Speech dataset](https://huggingface.co/datasets/MLCommons/peoples_speech):

```python
df = pd.read_parquet("hf://datasets/MLCommons/peoples_speech/microset/train-00000-of-00001.parquet")
df["audio"] = df["audio"].sf.enable()
```

You can also use the `datasets` library, here is an example on the [jlvdoorn/atco2-asr](https://huggingface.co/datasets/jlvdoorn/atco2-asr) dataset for automatic speech recognition:

```python
from datasets import load_dataset

df = load_dataset("jlvdoorn/atco2-asr", split="train").to_pandas()
df["audio"] = df["audio"].sf.enable()
```

Datasets created with `pandas-audio-methods` and saved to Parquet are also compatible with the [Dataset Viewer](https://huggingface.co/docs/hub/en/datasets-viewer) on Hugging Face and the [datasets](https://github.com/huggingface/datasets) library:

```python
df.to_parquet("hf://datasets/username/dataset_name/train.parquet")
```
