import inspect
import os
from functools import partialmethod
from io import BytesIO

import fsspec
import numpy as np
import pandas as pd
import soundfile as sf
import pyarrow as pa
from pandas._typing import Dtype
from pandas.api.extensions import ExtensionArray

from . import dask
from . import huggingface


dask.init()
huggingface.init()


def _audio_to_bytes(audio: sf.SoundFile) -> bytes:
    """Convert a sf.SoundFile object to bytes using native compression if possible, otherwise use PNG/TIFF compression."""
    buffer = BytesIO()
    offset = audio.tell()
    sf.write(buffer, audio.read(), samplerate=audio.samplerate, subtype=audio.subtype, endian=audio.endian, format=audio.format, compression_level=audio.compression_level, bitrate_mode=audio.bitrate_mode)
    audio.seek(offset)
    return buffer.getvalue()


def _encode_audio(audio: sf.SoundFile) -> dict:
    return {"bytes": _audio_to_bytes(audio), "path": audio.name if isinstance(audio.name, str) else None}


def _decode_audio(encoded_audio: dict) -> "sf.SoundFile":
    return sf.SoundFile(BytesIO(encoded_audio["bytes"])) if encoded_audio["bytes"] else sf.SoundFile(encoded_audio["path"])


class AudioArray(ExtensionArray):
    _pa_type = pa.struct({"bytes": pa.binary(), "path": pa.string()})

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    @property
    def dtype(self):
        dtype = pd.core.dtypes.dtypes.NumpyEADtype("object")
        dtype.construct_array_type = lambda: AudioArray
        return dtype

    @property
    def nbytes(self):
        return sum(audio.frames * 8 for audio in self)

    @property
    def feature(self):
        return {"_type": "Audio"}

    @classmethod
    def _from_sequence_of_strings(cls, strings, *, dtype: Dtype | None = None, copy: bool = False):
        a = np.empty(len(strings), dtype=object)
        a[:] = [sf.SoundFile(path if os.path.isfile(path) else fsspec.open(path).open()) if path is not None else None for path in strings]
        return cls(a)

    @classmethod
    def _from_sequence_of_audios(cls, audios, *, dtype: Dtype | None = None, copy: bool = False):
        a = np.empty(len(audios), dtype=object)
        a[:] = audios
        return cls(a)

    @classmethod
    def _from_sequence_of_encoded_audios(cls, encoded_audios, *, dtype: Dtype | None = None, copy: bool = False):
        a = np.empty(len(encoded_audios), dtype=object)
        a[:] = [_decode_audio(encoded_audio) if encoded_audio is not None else None for encoded_audio in encoded_audios]
        return cls(a)

    @classmethod
    def _from_sequence_of_data_and_samplerates(cls, data_and_samplerates, *, dtype: Dtype | None = None, copy: bool = False):
        a = np.empty(len(data_and_samplerates), dtype=object)
        for i, data_and_samplerate in enumerate(data_and_samplerates):
            if data_and_samplerate is None:
                a[i] = None
            else:
                data, samplerate = data_and_samplerate
                buffer = BytesIO()
                sf.write(buffer, data, samplerate, format="wav")
                buffer.seek(0)
                a[i] = sf.SoundFile(buffer)
        return cls(a)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        if len(scalars) == 0:
            return cls(np.array([], dtype=object))
        if isinstance(scalars[0], str):
            return cls._from_sequence_of_strings(scalars, dtype=dtype, copy=copy)
        if isinstance(scalars[0], dict) and set(scalars[0]) == {"bytes", "path"}:
            return cls._from_sequence_of_encoded_audios(scalars, dtype=dtype, copy=copy)
        elif isinstance(scalars[0], sf.SoundFile):
            return cls._from_sequence_of_audios(scalars, dtype=dtype, copy=copy)
        elif isinstance(scalars[0], tuple) and isinstance(scalars[0][0], np.ndarray) and isinstance(scalars[0][1], int):
            return cls._from_sequence_of_data_and_samplerates(scalars, dtype=dtype, copy=copy)
        raise TypeError(type(scalars[0].__name__))

    def __eq__(self, value: object) -> bool:
        return self.data == value.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> "sf.SoundFile | AudioArray":
        if isinstance(item, int):
            return self.data[item]
        return type(self)(self.data[item])

    def copy(self) -> "AudioArray":
        return AudioArray(self.data.copy())

    def __arrow_array__(self, type=None):
        return pa.array([_encode_audio(audio) if audio is not None else None for audio in self.data], type=self._pa_type)
    
    @classmethod
    def _empty(cls, shape, dtype=None):
        return cls(np.array([None] * shape, dtype=object))
    
    @classmethod
    def _concat_same_type(cls, to_concat):
        return AudioArray._from_sequence([audio for array in to_concat for audio in array])
    
    def take(self, indices, *, allow_fill=False, fill_value=None):
        from pandas.api.extensions import take

        if allow_fill and fill_value is None:
            fill_value = self.dtype.na_value

        result = take(self.data, indices, fill_value=fill_value, allow_fill=allow_fill)
        return self._from_sequence(result, dtype=self.dtype)

    def __reduce__(self):
        return AudioArray._from_sequence_of_encoded_audios, ([_encode_audio(audio) if audio is not None else None for audio in self.data],)


AudioArray._array_empty = AudioArray._empty(0)
AudioArray._array_nonempty = AudioArray._from_sequence_of_audios([sf.SoundFile(BytesIO(), "w", format="wav", samplerate=44000, channels=1)] * 2)


class SFMethods:
    _meta = pd.Series(AudioArray._array_empty)
    _meta_nonempty = pd.Series(AudioArray._array_nonempty)

    def __init__(self, data: pd.Series) -> None:
        self.data = data
    
    @classmethod
    def sf_method(cls, data, *, func, args, kwargs):
        return func(cls(data), *args, **kwargs)

    @dask.wrap_method
    def open(self):
        return pd.Series(AudioArray._from_sequence_of_strings(self.data))

    @dask.wrap_method
    def write(self):
        return pd.Series(AudioArray._from_sequence_of_data_and_samplerates(self.data))

    @dask.wrap_method
    def enable(self):
        return pd.Series(AudioArray._from_sequence(self.data))

    @dask.wrap_method
    def _apply(self, *args, _func, **kwargs):
        if not isinstance(self.data.array, AudioArray):
            raise Exception("You need to enable soundfile methods first, using for example: df['audio'] = df['audio'].sf.enable()")
        out = [_func(x, *args, **kwargs) for x in self.data]
        try:
            return pd.Series(type(self.data.array)._from_sequence(out))
        except TypeError:
            return pd.Series(out)


for _name, _func in inspect.getmembers(sf.SoundFile, predicate=inspect.isfunction):
    if not _name.startswith("_") and _name not in ["write"]:
        setattr(SFMethods, _name, partialmethod(SFMethods._apply, _func=_func))


_sts = pd.Series.to_string
def _new_sts(self, *args, **kwargs):
    return _sts(self, *args, **kwargs) + (", soundfile methods enabled" if isinstance(self.array, AudioArray) else "")
    
pd.Series.to_string = _new_sts
