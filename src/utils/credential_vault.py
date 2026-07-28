from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2
TARGET_PREFIX = "AxiControl/JARVIS/"


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", wintypes.FILETIME),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


class CredentialVault:
    """Minimal Windows Credential Manager wrapper for API secrets."""

    def __init__(self, prefix: str = TARGET_PREFIX):
        self.prefix = prefix
        self.available = os.name == "nt"
        if self.available:
            self._advapi = ctypes.WinDLL("Advapi32.dll")
            self._advapi.CredWriteW.argtypes = [
                ctypes.POINTER(_Credential),
                wintypes.DWORD,
            ]
            self._advapi.CredWriteW.restype = wintypes.BOOL
            self._advapi.CredReadW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(ctypes.POINTER(_Credential)),
            ]
            self._advapi.CredReadW.restype = wintypes.BOOL
            self._advapi.CredFree.argtypes = [ctypes.c_void_p]
            self._advapi.CredDeleteW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.DWORD,
                wintypes.DWORD,
            ]
            self._advapi.CredDeleteW.restype = wintypes.BOOL

    def _target(self, key: str) -> str:
        return self.prefix + str(key).strip().casefold()

    def set(self, key: str, value: str) -> bool:
        if not self.available:
            return False
        data = str(value).encode("utf-16-le")
        blob = (ctypes.c_ubyte * max(1, len(data)))()
        if data:
            ctypes.memmove(blob, data, len(data))
        credential = _Credential()
        credential.Type = CRED_TYPE_GENERIC
        credential.TargetName = self._target(key)
        credential.CredentialBlobSize = len(data)
        credential.CredentialBlob = ctypes.cast(
            blob, ctypes.POINTER(ctypes.c_ubyte)
        )
        credential.Persist = CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "AxiControl"
        return bool(self._advapi.CredWriteW(ctypes.byref(credential), 0))

    def get(self, key: str) -> str:
        if not self.available:
            return ""
        pointer = ctypes.POINTER(_Credential)()
        if not self._advapi.CredReadW(
            self._target(key), CRED_TYPE_GENERIC, 0, ctypes.byref(pointer)
        ):
            return ""
        try:
            credential = pointer.contents
            if not credential.CredentialBlob or not credential.CredentialBlobSize:
                return ""
            data = ctypes.string_at(
                credential.CredentialBlob, credential.CredentialBlobSize
            )
            return data.decode("utf-16-le")
        finally:
            self._advapi.CredFree(pointer)

    def delete(self, key: str) -> bool:
        if not self.available:
            return False
        return bool(
            self._advapi.CredDeleteW(
                self._target(key), CRED_TYPE_GENERIC, 0
            )
        )
