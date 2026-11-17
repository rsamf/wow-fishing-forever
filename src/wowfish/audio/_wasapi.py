"""Windows: capture one process's audio through WASAPI process loopback (Windows 10 2004+).

`ActivateAudioInterfaceAsync` on the virtual device "VAD\\Process_Loopback" with
AUDIOCLIENT_ACTIVATION_PARAMS naming the target process returns an IAudioClient that hears that
process tree alone. The client is initialised in shared loopback mode with the format the caller
chooses (the system converts), and packets are pulled from IAudioCaptureClient. Everything is
plain ctypes: no COM library is needed, so nothing changes for packaging. Only `_ComLoopback`
touches Windows; the layouts, the sample conversion and the read loop run anywhere.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import POINTER, Structure, Union, byref, c_ubyte, c_uint16, c_uint32, c_ulong, c_void_p

import numpy as np

from wowfish.audio import AudioUnavailable

# -- layouts (audioclientactivationparams.h, mmreg.h, propidl.h) ----------------------------------

AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_BUFFERFLAGS_SILENT = 0x2
VT_BLOB = 65
WAVE_FORMAT_PCM = 1
CAPTURE_RATE = 44100  # the Microsoft ApplicationLoopback sample's format; known to be accepted
CAPTURE_CHANNELS = 2
CAPTURE_BITS = 16
S_OK = 0
E_NOINTERFACE = 0x80004002
RPC_E_CHANGED_MODE = 0x80010106


class GUID(Structure):
    _fields_ = [
        ("Data1", c_uint32),
        ("Data2", c_uint16),
        ("Data3", c_uint16),
        ("Data4", c_ubyte * 8),
    ]

    @classmethod
    def from_text(cls, text: str) -> GUID:
        parts = text.strip("{}").split("-")
        g = cls()
        g.Data1 = int(parts[0], 16)
        g.Data2 = int(parts[1], 16)
        g.Data3 = int(parts[2], 16)
        tail = bytes.fromhex(parts[3] + parts[4])
        g.Data4 = (c_ubyte * 8)(*tail)
        return g

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GUID) and bytes(self) == bytes(other)

    __hash__ = None  # type: ignore[assignment]


IID_IUnknown = GUID.from_text("00000000-0000-0000-C000-000000000046")
IID_IAgileObject = GUID.from_text("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")
IID_IAudioClient = GUID.from_text("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_IAudioCaptureClient = GUID.from_text("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
IID_IActivateAudioInterfaceCompletionHandler = GUID.from_text(
    "41D949AB-9862-444A-80F6-C261334DA5EB"
)


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(Structure):
    _fields_ = [("TargetProcessId", c_uint32), ("ProcessLoopbackMode", c_uint32)]


class _ActivationUnion(Union):
    _fields_ = [("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]


class AUDIOCLIENT_ACTIVATION_PARAMS(Structure):
    _anonymous_ = ("u",)
    _fields_ = [("ActivationType", c_uint32), ("u", _ActivationUnion)]


class BLOB(Structure):
    _fields_ = [("cbSize", c_ulong), ("pBlobData", c_void_p)]


class _PropVariantUnion(Union):
    _fields_ = [("blob", BLOB), ("pad", c_ubyte * 16)]


class PROPVARIANT(Structure):
    _fields_ = [
        ("vt", c_uint16),
        ("wReserved1", c_uint16),
        ("wReserved2", c_uint16),
        ("wReserved3", c_uint16),
        ("u", _PropVariantUnion),
    ]


class WAVEFORMATEX(Structure):
    _pack_ = 1
    _fields_ = [
        ("wFormatTag", c_uint16),
        ("nChannels", c_uint16),
        ("nSamplesPerSec", c_uint32),
        ("nAvgBytesPerSec", c_uint32),
        ("nBlockAlign", c_uint16),
        ("wBitsPerSample", c_uint16),
        ("cbSize", c_uint16),
    ]


def activation_params(pid: int) -> AUDIOCLIENT_ACTIVATION_PARAMS:
    p = AUDIOCLIENT_ACTIVATION_PARAMS()
    p.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
    p.ProcessLoopbackParams.TargetProcessId = pid
    p.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
    return p


def capture_format() -> WAVEFORMATEX:
    f = WAVEFORMATEX()
    f.wFormatTag = WAVE_FORMAT_PCM
    f.nChannels = CAPTURE_CHANNELS
    f.nSamplesPerSec = CAPTURE_RATE
    f.wBitsPerSample = CAPTURE_BITS
    f.nBlockAlign = CAPTURE_CHANNELS * CAPTURE_BITS // 8
    f.nAvgBytesPerSec = CAPTURE_RATE * f.nBlockAlign
    f.cbSize = 0
    return f


def to_mono_22050(raw: bytes) -> np.ndarray:
    """44100 Hz stereo int16 frames -> 22050 Hz mono float32 (average channels, then pairs)."""
    pcm = np.frombuffer(raw, dtype="<i2")
    n = len(pcm) // (2 * CAPTURE_CHANNELS) * (2 * CAPTURE_CHANNELS)
    frames = pcm[:n].astype(np.float32).reshape(-1, CAPTURE_CHANNELS).mean(axis=1)
    return (frames.reshape(-1, 2).mean(axis=1) / 32768.0).astype(np.float32)


# -- the source -----------------------------------------------------------------------------------


class ProcessLoopbackSource:
    """AudioSource that hears one process. `backend` is swapped for a fake in tests."""

    def __init__(
        self,
        pid: int,
        *,
        sample_rate: int = 22050,
        blocksize: int = 1024,
        backend: object | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        if sample_rate != CAPTURE_RATE // 2:
            raise ValueError("ProcessLoopbackSource delivers 22050 Hz")
        self.pid = pid
        self.name = f"process {pid} (game audio only)"
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.timeout_s = timeout_s
        self._backend = backend if backend is not None else _ComLoopback(pid)
        self._pending = np.zeros(0, dtype=np.float32)
        self._leftover = b""

    def __enter__(self) -> ProcessLoopbackSource:
        self._backend.open()  # type: ignore[attr-defined]
        return self

    def __exit__(self, *exc: object) -> None:
        self._backend.close()  # type: ignore[attr-defined]

    def read(self, n_frames: int) -> np.ndarray:
        waited = 0.0
        while len(self._pending) < n_frames:
            packet = self._backend.read_packet()  # type: ignore[attr-defined]
            if packet is None:
                if waited >= self.timeout_s:
                    raise RuntimeError(f"no audio from process {self.pid} for {self.timeout_s:g} s")
                time.sleep(0.005)
                waited += 0.005
                continue
            waited = 0.0
            raw = self._leftover + packet
            usable = len(raw) // 8 * 8  # two stereo int16 frames per output sample
            self._leftover = raw[usable:]
            self._pending = np.concatenate([self._pending, to_mono_22050(raw[:usable])])
        out, self._pending = self._pending[:n_frames], self._pending[n_frames:]
        return np.ascontiguousarray(out, dtype=np.float32)


# -- COM, Windows only ----------------------------------------------------------------------------


def _check(hr: int, what: str) -> None:
    if hr != S_OK:
        raise AudioUnavailable(f"{what} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")


def _method(obj: int, index: int, restype: object, *argtypes: object):  # type: ignore[no-untyped-def]
    """The `index`-th virtual method of the COM interface pointer `obj`."""
    vtable = ctypes.cast(ctypes.c_void_p(obj), POINTER(POINTER(c_void_p)))[0]
    proto = ctypes.WINFUNCTYPE(restype, c_void_p, *argtypes)  # type: ignore[attr-defined]
    return proto(vtable[index])


class _ComLoopback:
    """Owns the activation handler, IAudioClient and IAudioCaptureClient for one process."""

    def __init__(self, pid: int) -> None:
        if sys.platform != "win32":
            raise AudioUnavailable("WASAPI process loopback is Windows-only")
        self.pid = pid
        self._client = 0
        self._capture = 0
        self._event = None
        self._keepalive: list[object] = []

    # The completion handler is a COM object built by hand: a struct whose first member points at
    # a vtable of four callbacks (IUnknown's three, then ActivateCompleted).
    def _make_handler(self, done: threading.Event):  # type: ignore[no-untyped-def]
        wintypes = ctypes.wintypes
        QI = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, POINTER(GUID), POINTER(c_void_p))  # type: ignore[attr-defined]
        REF = ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)  # type: ignore[attr-defined]
        DONE = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, c_void_p)  # type: ignore[attr-defined]

        class VTable(Structure):
            _fields_ = [
                ("QueryInterface", QI),
                ("AddRef", REF),
                ("Release", REF),
                ("ActivateCompleted", DONE),
            ]

        class Handler(Structure):
            _fields_ = [("lpVtbl", POINTER(VTable))]

        accepted = (IID_IUnknown, IID_IAgileObject, IID_IActivateAudioInterfaceCompletionHandler)

        def query_interface(this: int, riid, ppv) -> int:  # type: ignore[no-untyped-def]
            if any(riid[0] == iid for iid in accepted):
                ppv[0] = this
                return S_OK
            ppv[0] = None
            return ctypes.c_int32(E_NOINTERFACE).value

        def activate_completed(this: int, op: int) -> int:
            done.set()
            return S_OK

        vtable = VTable(
            QI(query_interface), REF(lambda this: 1), REF(lambda this: 1), DONE(activate_completed)
        )
        handler = Handler(ctypes.pointer(vtable))
        self._keepalive += [vtable, handler, query_interface, activate_completed]
        return handler

    def open(self) -> None:
        ole32 = ctypes.windll.ole32  # type: ignore[attr-defined]
        mmdevapi = ctypes.windll.Mmdevapi  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        hr = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        if hr not in (S_OK, 1) and (hr & 0xFFFFFFFF) != RPC_E_CHANGED_MODE:
            _check(hr, "CoInitializeEx")

        params = activation_params(self.pid)
        prop = PROPVARIANT()
        prop.vt = VT_BLOB
        prop.u.blob.cbSize = ctypes.sizeof(params)
        prop.u.blob.pBlobData = ctypes.cast(byref(params), c_void_p)
        done = threading.Event()
        handler = self._make_handler(done)
        op = c_void_p()
        mmdevapi.ActivateAudioInterfaceAsync.restype = ctypes.HRESULT
        _check(
            mmdevapi.ActivateAudioInterfaceAsync(
                ctypes.c_wchar_p(VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK),
                byref(IID_IAudioClient),
                byref(prop),
                byref(handler),
                byref(op),
            ),
            "ActivateAudioInterfaceAsync (hearing one app needs Windows 10 version 2004 or later)",
        )
        if not done.wait(5.0):
            raise AudioUnavailable("ActivateAudioInterfaceAsync did not complete")
        result_hr = ctypes.c_int32()
        unknown = c_void_p()
        _check(
            _method(op.value, 3, ctypes.HRESULT, POINTER(ctypes.c_int32), POINTER(c_void_p))(
                op.value, byref(result_hr), byref(unknown)
            ),
            "GetActivateResult",
        )
        _check(result_hr.value, "activating the process loopback client")
        _method(op.value, 2, c_ulong)(op.value)  # Release the async operation
        self._client = unknown.value or 0

        fmt = capture_format()
        _check(
            _method(
                self._client,
                3,
                ctypes.HRESULT,
                c_uint32,
                c_uint32,
                ctypes.c_int64,
                ctypes.c_int64,
                POINTER(WAVEFORMATEX),
                c_void_p,
            )(
                self._client,
                AUDCLNT_SHAREMODE_SHARED,
                AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                2_000_000,  # 200 ms buffer, in 100 ns units
                0,
                byref(fmt),
                None,
            ),
            "IAudioClient::Initialize",
        )
        self._event = kernel32.CreateEventW(None, False, False, None)
        _check(
            _method(self._client, 13, ctypes.HRESULT, c_void_p)(self._client, self._event),
            "IAudioClient::SetEventHandle",
        )
        capture = c_void_p()
        _check(
            _method(self._client, 14, ctypes.HRESULT, POINTER(GUID), POINTER(c_void_p))(
                self._client, byref(IID_IAudioCaptureClient), byref(capture)
            ),
            "IAudioClient::GetService(IAudioCaptureClient)",
        )
        self._capture = capture.value or 0
        _check(_method(self._client, 10, ctypes.HRESULT)(self._client), "IAudioClient::Start")

    def read_packet(self) -> bytes | None:
        """One capture packet as raw 44100 Hz stereo int16 bytes, or None when nothing arrived."""
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.WaitForSingleObject(self._event, 100)
        frames = c_uint32()
        _check(
            _method(self._capture, 5, ctypes.HRESULT, POINTER(c_uint32))(
                self._capture, byref(frames)
            ),
            "IAudioCaptureClient::GetNextPacketSize",
        )
        if frames.value == 0:
            return None
        data = c_void_p()
        got = c_uint32()
        flags = c_uint32()
        _check(
            _method(
                self._capture,
                3,
                ctypes.HRESULT,
                POINTER(c_void_p),
                POINTER(c_uint32),
                POINTER(c_uint32),
                c_void_p,
                c_void_p,
            )(self._capture, byref(data), byref(got), byref(flags), None, None),
            "IAudioCaptureClient::GetBuffer",
        )
        n_bytes = got.value * CAPTURE_CHANNELS * CAPTURE_BITS // 8
        if flags.value & AUDCLNT_BUFFERFLAGS_SILENT or not data.value:
            packet = bytes(n_bytes)
        else:
            packet = ctypes.string_at(data.value, n_bytes)
        _check(
            _method(self._capture, 4, ctypes.HRESULT, c_uint32)(self._capture, got.value),
            "IAudioCaptureClient::ReleaseBuffer",
        )
        return packet

    def close(self) -> None:
        if self._client:
            _method(self._client, 11, ctypes.HRESULT)(self._client)  # Stop
        for ptr in (self._capture, self._client):
            if ptr:
                _method(ptr, 2, c_ulong)(ptr)  # Release
        self._capture = self._client = 0
        if self._event:
            ctypes.windll.kernel32.CloseHandle(self._event)  # type: ignore[attr-defined]
            self._event = None
