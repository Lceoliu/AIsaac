"""Reader for the game's packed resource archives (`resources/packed/*.a`, magic ``ARCH000``).

Translated from the Afterbirth+ v1.06 Linux build (named symbols, see
analysis/abplus-linux/exports/abplus-x64-baseline):

- name hashes      KAGE::Filesys::FileManager::HashDJB2 / HashFNV1          (0x46BFF0 / 0x46C050)
- index layout     KAGE::Filesys::FileManager::LoadArchiveFile               (0x46C560)
- lookup           KAGE::Filesys::FileManager::get_archived_file             (0x46C150)
- entry decoding   KAGE::Filesys::ArchivedFile::rewind / refill_buffer       (0x48B5E0 / 0x48B750)
                   MiniZ::decode / scramble, ISAACRNG                        (0x4ADC80 / 0x4ADB40)
- checksum         KAGE::Filesys::ArchivedFile::IsChecksumValid              (0x48B450)

Gibbed.Rebirth.Unpack only knows names from its own file list and crashes on some unnamed entries;
hashing the path directly finds every file the engine can open (e.g. the Afterbirth alt-floor room
files ``resources/rooms/03.burning basement.stb``). Later archives override earlier ones for the
same (hashA, hashB), as in LoadArchiveFile.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

MASK32 = 0xFFFFFFFF


def _normalise(path: str) -> bytes:
    out = bytearray()
    for ch in path.encode('latin-1'):
        if 0x41 <= ch <= 0x5A:
            ch += 0x20
        elif ch == 0x5C:
            ch = 0x2F
        out.append(ch)
    return bytes(out)


def hash_djb2(path: str) -> int:
    h = 0x1505
    for c in _normalise(path):
        h = (c + h * 0x21) & MASK32
    return h


def hash_fnv1(path: str, h: int = 0x5BB2220E) -> int:
    for c in _normalise(path):
        h = ((h ^ c) * 0x1000193) & MASK32
    return h


def checksum(data: bytes) -> int:
    """IsChecksumValid: rotate right by 1, then add each little-endian dword.

    The engine reads 0x200 bytes at a time into one reused (initially zeroed) buffer and rounds the
    last read up to whole dwords, so a trailing partial dword picks up stale bytes left over from
    the previous read, not zeros.
    """
    h = 0xABABEB98
    buf = bytearray(0x200)
    for start in range(0, len(data), 0x200):
        chunk = data[start:start + 0x200]
        buf[:len(chunk)] = chunk
        for (word,) in struct.iter_unpack('<I', bytes(buf[:(len(chunk) + 3) & ~3])):
            h = (((h << 31) | (h >> 1)) + word) & MASK32
    return h


class IsaacRng:
    """Bob Jenkins' ISAAC as used by MiniZ::scramble for stored (uncompressed) archive blocks.

    Seeded with a single 32-bit value (the entry's FNV hash). Not needed by any room file seen so far;
    the stored path raises until it is verified against a real entry.
    """

    def __init__(self, seed: int):
        raise NotImplementedError('stored MiniZ blocks (ISAACRNG) are not translated yet')


@dataclass(frozen=True)
class Entry:
    archive: 'Archive'
    hash_a: int
    hash_b: int
    offset: int
    length: int
    checksum: int

    def read(self) -> bytes:
        return self.archive.read_entry(self)


class Archive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, 'rb') as f:
            head = f.read(14)
            if head[:7] != b'ARCH000':
                raise ValueError(f'{self.path}: not an ARCH000 archive')
            self.kind = head[7]
            index_offset, count = struct.unpack_from('<IH', head, 8)
            f.seek(index_offset)
            raw = f.read(count * 20)
        self.entries: dict[tuple[int, int], Entry] = {}
        for i in range(count):
            a, b, off, length, crc = struct.unpack_from('<5I', raw, i * 20)
            self.entries[(a, b)] = Entry(self, a, b, off, length, crc)

    def find(self, name: str) -> Entry | None:
        return self.entries.get((hash_djb2(name), hash_fnv1(name)))

    def read_entry(self, entry: Entry) -> bytes:
        with open(self.path, 'rb') as f:
            f.seek(entry.offset)
            if self.kind == 2:
                data = self._read_miniz(f, entry)
            elif self.kind in (0, 3, 4):
                data = self._read_scrambled(f, entry)
            else:
                raise NotImplementedError(f'archive kind {self.kind}')
        data = data[:entry.length]
        if checksum(data) != entry.checksum:
            raise ValueError(f'{self.path.name}: checksum mismatch for entry {entry.hash_a:08x}{entry.hash_b:08x}')
        return data

    @staticmethod
    def _read_miniz(f, entry: Entry) -> bytes:
        # refill_buffer kind 2: [u32 header: bit31 = final, low 31 bits = compressed size] + bytes.
        # A non-final chunk of exactly 0x400 bytes switches the stream to stored+scrambled mode.
        inflater = zlib.decompressobj(-15)
        out = bytearray()
        while len(out) < entry.length:
            (header,) = struct.unpack('<I', f.read(4))
            size, final = header & 0x7FFFFFFF, bool(header >> 31)
            chunk = f.read(size)
            if not final and size == 0x400:
                IsaacRng(entry.hash_b)
            out += inflater.decompress(chunk)
            if final:
                out += inflater.flush()
                break
        return bytes(out)

    @staticmethod
    def _read_scrambled(f, entry: Entry) -> bytes:
        # rewind: key = hashB ^ 0xF9524287 | 1; unscramble dword by dword, re-keyed per 0x400 block.
        key = (entry.hash_b ^ 0xF9524287) | 1
        out = bytearray()
        while len(out) < entry.length:
            want = min(0x400, (entry.length + 3 - len(out)) & ~3)
            block = bytearray(f.read(want))
            key = _unscramble(block, key)
            out += block
        return bytes(out)


def _xorshift_key(k: int) -> int:
    k ^= (k << 8) & MASK32
    k ^= k >> 9
    return (k ^ (k << 23)) & MASK32


def _unscramble(buf: bytearray, key: int) -> int:
    """ArchivedFile::unscramble; returns the key for the next block (refill_buffer stores it)."""
    for i in range(0, len(buf) - len(buf) % 4, 4):
        word = struct.unpack_from('<I', buf, i)[0] ^ key
        b = bytearray(struct.pack('<I', word))
        low = key & 0xF
        if low == 9:
            b[0], b[1], b[2], b[3] = b[1], b[0], b[3], b[2]
        elif low == 13:
            b[0], b[1], b[2], b[3] = b[2], b[3], b[0], b[1]
        elif low == 2:
            b[0], b[1], b[2], b[3] = b[3], b[2], b[1], b[0]
        buf[i:i + 4] = b
        key = _xorshift_key(key)
    return key


class ArchiveSet:
    """Archives in load order; a later archive overrides an earlier one (LoadArchiveFile)."""

    def __init__(self, paths):
        self.archives = [Archive(p) for p in paths]

    def find(self, name: str) -> Entry | None:
        for archive in reversed(self.archives):
            entry = archive.find(name)
            if entry is not None:
                return entry
        return None

    def read(self, name: str) -> bytes:
        entry = self.find(name)
        if entry is None:
            raise FileNotFoundError(name)
        return entry.read()
