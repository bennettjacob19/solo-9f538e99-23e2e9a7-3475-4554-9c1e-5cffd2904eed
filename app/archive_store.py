"""归档槽位：两个可回退的检查点，内容寻址分块去重。

存储布局（DATA_DIR/archive/）::

    state.json                 归档容量、槽位描述、操作历史
    store/<前2位>/<sha256>      数组数据块（4 KiB 定长，内容寻址）
    archives/<id>/index.json   一份留档的数组块索引 + 训练步等元数据
    exports/<id>/rank_*.npz    从留档在线重分片出的下载文件（只保留最近一次）

设计要点：

- 参数 / Adam 一阶 / 二阶三个数组按 **全局字节流** 每 4096 字节切块，
  块名为内容 sha256；数组和 rank 切法无关，训练步只放在元数据里。
  因此只改训练步或只改 rank 切法的两版共享全部数组数据块。
- 一段连续 B 字节的变化最多弄脏 ``ceil(B/4096)+1`` 个块，
  新增数组数据不超过 ``B + 2*4096`` 字节，即 ``B + 8 KiB``。
  占用统计只算数组数据块，不含 index/manifest 元数据。
- 归档容量只按去重后的数据块字节数计；建归档先算新增字节，超额或写失败
  时删除本次新写的块，已有留档、容量与占用都不变。
- 删除留档后重新计算仍被保留留档引用的块，立即删掉无人引用的块；
  保留留档的 index 和数据块完全不动。
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from reshard import (ARRAY_KEYS, ReshardError, split_offsets, split_sizes,
                     write_shard)

CHUNK_SIZE = 4096
MAX_ARCHIVES = 2


class ArchiveError(ReshardError):
    """归档失败基类，message 面向用户。"""


class ArchiveSlotFullError(ArchiveError):
    """两个回退位都已占用。"""


class ArchiveCapacityError(ArchiveError):
    """归档容量超限。"""


class ArchiveWriteError(ArchiveError):
    """归档写失败（真实 IO 错误或模拟写失败）。"""


class ArchiveMissingError(ArchiveError):
    """留档或留档数据块不存在 / 校验失败。"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _dir_bytes(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_chunks(raw: bytes):
    """按固定 4 KiB 切块。"""
    for off in range(0, len(raw), CHUNK_SIZE):
        yield raw[off:off + CHUNK_SIZE]


class ArchiveStore:
    """两个回退位的去重内容存储。"""

    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir) / "archive"
        self.store_root = self.dir / "store"
        self.archives_root = self.dir / "archives"
        self.exports_root = self.dir / "exports"
        self.state_file = self.dir / "state.json"
        for d in (self.store_root, self.archives_root, self.exports_root):
            d.mkdir(parents=True, exist_ok=True)
        self._state = self._load()
        # 进程在“写块后、写状态前”崩溃会留下无人引用的块；启动时清掉，
        # 保证页面上的占用就是留档真正保留的数组数据。
        self.reconcile()

    def reconcile(self) -> int:
        """删除没有任何留档引用的数据块（保留留档的索引与块不动）。"""
        try:
            referenced = self._referenced_hashes()
        except ArchiveError:
            return 0  # 索引损坏时宁可不清理
        removed = 0
        for path in list(self.store_root.rglob("*")):
            if path.is_file() and path.name not in referenced:
                removed += path.stat().st_size
                path.unlink(missing_ok=True)
        for d in sorted((p for p in self.store_root.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        return removed

    # ------------------------------------------------------------------ 状态
    def _load(self) -> dict:
        if self.state_file.is_file():
            try:
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                state.setdefault("capacity_bytes", 1 << 20)
                state.setdefault("archives", [])
                state.setdefault("history", [])
                return state
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return {
            "capacity_bytes": 1 << 20,  # 默认 1 MiB
            "archives": [],
            "history": [],
        }

    def save(self) -> None:
        tmp = self.state_file.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(self.state_file)

    @property
    def capacity(self) -> int:
        return int(self._state["capacity_bytes"])

    @capacity.setter
    def capacity(self, value: int) -> None:
        value = int(value)
        if value < 0:
            raise ArchiveError("归档容量必须 >= 0")
        self._state["capacity_bytes"] = value
        self.save()

    @property
    def archives(self) -> list[dict]:
        return list(self._state["archives"])

    @property
    def history(self) -> list[dict]:
        return self._state["history"]

    def append_history(self, entry: dict) -> None:
        self._state["history"].append(entry)
        del self._state["history"][:-50]

    def stored_bytes(self) -> int:
        """去重后实际保留的数组数据块字节数（不含 index/manifest）。"""
        return _dir_bytes(self.store_root)

    def get_archive(self, archive_id: str) -> dict | None:
        try:
            archive_id = self._safe_archive_id(archive_id)
        except ArchiveMissingError:
            return None
        for desc in self._state["archives"]:
            if desc["id"] == archive_id:
                return desc
        return None

    def _archive_dir(self, archive_id: str) -> Path:
        return self.archives_root / archive_id

    @staticmethod
    def _safe_archive_id(archive_id: str) -> str:
        """留档 ID 来自表单，必须形如 arc-… 且不含路径分隔符。"""
        if (not archive_id or archive_id.startswith(".")
                or "/" in archive_id or "\\" in archive_id
                or archive_id != Path(archive_id).name):
            raise ArchiveMissingError(f"留档 ID 非法: {archive_id!r}")
        return archive_id

    def _index_path(self, archive_id: str) -> Path:
        return self._archive_dir(self._safe_archive_id(archive_id)) / "index.json"

    def _read_index(self, archive_id: str) -> dict:
        path = self._index_path(archive_id)
        if not path.is_file():
            raise ArchiveMissingError(f"留档 {archive_id} 的索引不存在")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArchiveMissingError(f"留档 {archive_id} 的索引损坏: {exc}") from exc

    # ------------------------------------------------------------------ 数据块
    def _chunk_path(self, digest: str) -> Path:
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ArchiveMissingError(f"数据块名非法: {digest!r}")
        return self.store_root / digest[:2] / digest

    @staticmethod
    def _chunk_hashes(index: dict) -> list[str]:
        out: list[str] = []
        for k in ARRAY_KEYS:
            out.extend(index["arrays"][k]["chunks"])
        return out

    def _archive_hashes(self, archive_id: str) -> list[str]:
        return self._chunk_hashes(self._read_index(archive_id))

    def exclusive_bytes(self, archive_id: str) -> int:
        """仅该留档引用、删掉它会释放的数组数据字节数。"""
        counts: Counter[str] = Counter()
        sizes: dict[str, int] = {}
        for desc in self._state["archives"]:
            idx = self._read_index(desc["id"])
            counts.update(self._chunk_hashes(idx))
        own = set(self._archive_hashes(archive_id))
        for digest in own:
            p = self._chunk_path(digest)
            sizes[digest] = p.stat().st_size if p.is_file() else 0
        return sum(sizes[d] for d in own if counts[d] == 1)

    def page_stats(self) -> dict:
        """页面用的容量 / 占用 / 槽位摘要。"""
        archives = []
        for desc in self._state["archives"]:
            item = dict(desc)
            try:
                item["exclusive_bytes"] = self.exclusive_bytes(desc["id"])
            except ArchiveError:
                item["exclusive_bytes"] = None
            archives.append(item)
        logical = sum(int(a.get("logical_array_bytes", 0)) for a in archives)
        return {
            "capacity": self.capacity,
            "used_bytes": self.stored_bytes(),
            "logical_bytes": logical,
            "slots_used": len(self._state["archives"]),
            "max_slots": MAX_ARCHIVES,
            "archives": archives,
        }

    # ------------------------------------------------------------------ 建归档
    def create_archive(self, published: dict, arrays: dict[str, np.ndarray],
                       simulate_write_fail: bool = False) -> dict:
        """把当前发布的一套全局数组写成留档。

        ``arrays`` 必须是三个 1-D 全局数组（与 rank 切法无关）。失败时
        不改变已有留档、归档容量和占用。
        """
        if len(self._state["archives"]) >= MAX_ARCHIVES:
            raise ArchiveSlotFullError(
                f"两个回退位都已占用（{len(self._state['archives'])}/{MAX_ARCHIVES}），"
                "请先删除任一留档"
            )
        if published.get("requires_old_volume"):
            raise ArchiveError("当前发布依赖旧卷，不能存档；仅可存档不依赖旧卷的发布")

        flat: dict[str, np.ndarray] = {}
        for k in ARRAY_KEYS:
            if k not in arrays:
                raise ArchiveError(f"发布数据缺少数组 {k}")
            flat[k] = np.ascontiguousarray(arrays[k]).reshape(-1)
        dtypes = {str(flat[k].dtype) for k in ARRAY_KEYS}
        lengths = {int(flat[k].size) for k in ARRAY_KEYS}
        if len(dtypes) != 1:
            raise ArchiveError(f"三个数组 dtype 不一致: {sorted(dtypes)}")
        if len(lengths) != 1:
            raise ArchiveError(f"三个数组长度不一致: {sorted(lengths)}")
        total = lengths.pop()
        if int(published.get("total_elems", total)) != total:
            raise ArchiveError(
                f"发布记录的 total_elems={published.get('total_elems')} "
                f"与实际数组长度 {total} 不一致"
            )

        # 切块、去重；同时记录每个数组实际新增的字节数，便于核对增量占用。
        unique: dict[str, bytes] = {}
        array_index: dict[str, dict] = {}
        new_by_array: dict[str, int] = {k: 0 for k in ARRAY_KEYS}
        seen: set[str] = set()
        for k in ARRAY_KEYS:
            raw = flat[k].tobytes(order="C")
            chunks: list[str] = []
            for chunk in iter_chunks(raw):
                digest = _sha256(chunk)
                chunks.append(digest)
                if digest not in unique:
                    unique[digest] = chunk
                elif unique[digest] != chunk:
                    raise ArchiveError("归档数据块 sha256 冲突")
                path = self._chunk_path(digest)
                if digest not in seen and not path.exists():
                    new_by_array[k] += len(chunk)
                seen.add(digest)
            array_index[k] = {
                "dtype": str(flat[k].dtype),
                "length": int(flat[k].size),
                "nbytes": len(raw),
                "chunks": chunks,
            }

        before = self.stored_bytes()
        new_bytes = sum(len(b) for digest, b in unique.items()
                        if not self._chunk_path(digest).exists())
        if before + new_bytes > self.capacity:
            if before > self.capacity:
                detail = (f"已保留的数组数据 {before} B 本身就超过当前归档容量 "
                          f"{self.capacity} B（容量调小不会删除已有留档）")
            else:
                detail = (f"已保留 {before} B + 本次需新增 {new_bytes} B "
                          f"> 归档容量 {self.capacity} B")
            raise ArchiveCapacityError(f"归档容量超限: {detail}")

        archive_id = f"arc-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        created_at = _now()
        index = {
            "id": archive_id,
            "source_publish_id": published.get("id"),
            "source_ckpt": published.get("source_ckpt"),
            "source_ranks": int(published.get("ranks", 0)),
            "source_shard_sizes": list(published.get("shard_sizes") or []),
            "step": int(published["step"]),
            "dtype": dtypes.pop(),
            "total_elems": total,
            "logical_array_bytes": total * np.dtype(flat["param"].dtype).itemsize * len(ARRAY_KEYS),
            "new_data_bytes": new_bytes,
            "new_data_bytes_by_array": new_by_array,
            "created_at": created_at,
            "arrays": array_index,
        }

        created_chunks: list[str] = []
        archive_dir = self._archive_dir(archive_id)
        desc: dict | None = None
        try:
            to_write = [(h, b) for h, b in unique.items()
                        if not self._chunk_path(h).exists()]
            fail_at = len(to_write) // 2 if simulate_write_fail else -1
            for i, (digest, chunk) in enumerate(to_write):
                if i == fail_at:
                    raise ArchiveWriteError(
                        f"模拟归档写失败: 写数据块 {digest[:12]}… 时出错"
                    )
                self._write_chunk(digest, chunk)
                created_chunks.append(digest)
            if simulate_write_fail and not to_write:
                raise ArchiveWriteError("模拟归档写失败：写入留档元数据时出错")

            archive_dir.mkdir(parents=True, exist_ok=False)
            tmp_index = archive_dir / "index.json.tmp"
            tmp_index.write_text(json.dumps(index, ensure_ascii=False),
                                 encoding="utf-8")
            tmp_index.replace(archive_dir / "index.json")

            desc = {
                "id": archive_id,
                "source_publish_id": index["source_publish_id"],
                "source_ckpt": index["source_ckpt"],
                "source_ranks": index["source_ranks"],
                "source_shard_sizes": index["source_shard_sizes"],
                "step": index["step"],
                "dtype": index["dtype"],
                "total_elems": index["total_elems"],
                "logical_array_bytes": index["logical_array_bytes"],
                "new_data_bytes": index["new_data_bytes"],
                "new_data_bytes_by_array": index["new_data_bytes_by_array"],
                "created_at": created_at,
                "store_bytes_after": before + new_bytes,
            }
            self._state["archives"].append(desc)
            self.save()
            return desc
        except Exception:
            if desc is not None and desc in self._state["archives"]:
                self._state["archives"].remove(desc)
            shutil.rmtree(archive_dir, ignore_errors=True)
            self._remove_chunks(created_chunks)
            raise

    def _write_chunk(self, digest: str, data: bytes) -> None:
        path = self._chunk_path(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{digest}.{uuid.uuid4().hex}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _remove_chunks(self, digests: list[str]) -> None:
        for digest in digests:
            path = self._chunk_path(digest)
            path.unlink(missing_ok=True)
            try:
                path.parent.rmdir()  # 只删空目录
            except OSError:
                pass

    # ------------------------------------------------------------------ 删归档
    def delete_archive(self, archive_id: str) -> dict:
        """删留档；只剩保留留档引用的数据块。保留留档不重写。

        等被删留档独占的块删完、状态落盘后才算成功；中途失败时索引仍指回
        原状态，刷新页面会由 ``reconcile`` 清掉已删的残块。
        """
        desc = self.get_archive(archive_id)
        if desc is None:
            raise ArchiveMissingError(f"留档 {archive_id} 不存在")

        # 先把索引挪到回收目录：即使随后崩溃，启动时也不会把已删留档算作引用。
        trash = self.dir / f".trash-{uuid.uuid4().hex}"
        shutil.move(str(self._archive_dir(archive_id)), str(trash))
        old_list = self._state["archives"]
        self._state["archives"] = [d for d in old_list if d["id"] != archive_id]
        try:
            self.save()
        except Exception:
            self._state["archives"] = old_list
            shutil.move(str(trash), str(self._archive_dir(archive_id)))
            raise
        try:
            self._prune_unreferenced()
        finally:
            shutil.rmtree(trash, ignore_errors=True)
        return desc

    def _referenced_hashes(self) -> set[str]:
        referenced: set[str] = set()
        for desc in self._state["archives"]:
            referenced.update(self._archive_hashes(desc["id"]))
        return referenced

    def _prune_unreferenced(self) -> int | None:
        """删除无留档引用的块；返回释放字节数。

        保留留档的索引读不出来时返回 None 且不删任何块（宁可少释放，
        也不能删掉仍被引用的数据）。
        """
        try:
            referenced = self._referenced_hashes()
        except ArchiveError:
            return None
        freed = 0
        for path in list(self.store_root.rglob("*")):
            if path.is_file() and path.name not in referenced:
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
        for d in sorted((p for p in self.store_root.rglob("*") if p.is_dir()),
                        key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
        return freed

    # ------------------------------------------------------------------ 读归档
    def load_arrays(self, archive_id: str) -> dict:
        """仅从该留档读取三个全局数组与训练步。"""
        desc = self.get_archive(archive_id)
        if desc is None:
            raise ArchiveMissingError(f"留档 {archive_id} 不存在")
        index = self._read_index(archive_id)
        dtype = np.dtype(index["dtype"])
        arrays: dict[str, np.ndarray] = {}
        for k in ARRAY_KEYS:
            meta = index["arrays"][k]
            buf = bytearray()
            for digest in meta["chunks"]:
                path = self._chunk_path(digest)
                if not path.is_file():
                    raise ArchiveMissingError(
                        f"留档 {archive_id} 的数组 {k} 缺少数据块 {digest[:12]}…"
                    )
                data = path.read_bytes()
                if _sha256(data) != digest:
                    raise ArchiveMissingError(
                        f"留档 {archive_id} 的数组 {k} 数据块 {digest[:12]}… 校验失败"
                    )
                buf.extend(data)
            expected = int(meta["length"]) * dtype.itemsize
            if len(buf) != expected:
                raise ArchiveMissingError(
                    f"留档 {archive_id} 的数组 {k} 长度 {len(buf)} B，"
                    f"期望 {expected} B"
                )
            arrays[k] = np.frombuffer(bytes(buf), dtype=dtype).copy()
        lengths = {int(arrays[k].size) for k in ARRAY_KEYS}
        if len(lengths) != 1:
            raise ArchiveMissingError(f"留档三个数组长度不一致: {sorted(lengths)}")
        return {"step": int(index["step"]), "dtype": str(dtype),
                "arrays": arrays, "index": index}

    # ------------------------------------------------------------------ 导出
    def export_from_archive(self, archive_id: str, n_new: int) -> dict:
        """从留档重分片，生成可下载的 npz；只读该留档。"""
        if n_new < 1:
            raise ArchiveError("新 rank 数必须 >= 1")
        loaded = self.load_arrays(archive_id)
        arrays = loaded["arrays"]
        total = int(arrays["param"].size)
        sizes = split_sizes(total, n_new)
        offsets = split_offsets(total, n_new)
        step = int(loaded["step"])
        export_id = f"exp-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
        staging = self.dir / f".export-stg-{uuid.uuid4().hex}"
        staging.mkdir(parents=True, exist_ok=False)
        try:
            total_bytes = 0
            for r, (off, size) in enumerate(offsets):
                total_bytes += write_shard(
                    staging / f"rank_{r}.npz",
                    {k: arrays[k][off:off + size] for k in ARRAY_KEYS},
                    step,
                )
            # 回读校验：训练步、长度、内容都与留档一致才交付下载。
            for r, (off, size) in enumerate(offsets):
                with np.load(staging / f"rank_{r}.npz") as z:
                    if int(z["step"]) != step:
                        raise ArchiveError(
                            f"留档重分片回读训练步不一致: rank {r}"
                        )
                    for k in ARRAY_KEYS:
                        got = np.ascontiguousarray(z[k]).reshape(-1)
                        if got.size != size or not np.array_equal(
                                got, arrays[k][off:off + size]):
                            raise ArchiveError(
                                f"留档重分片回读内容不一致: rank {r} / {k}"
                            )
            manifest = {
                "export_id": export_id,
                "source_archive_id": archive_id,
                "source_publish_id": loaded["index"].get("source_publish_id"),
                "step": step,
                "dtype": loaded["dtype"],
                "total_elems": total,
                "ranks": n_new,
                "shard_sizes": sizes,
                "bytes": total_bytes,
                "created_at": _now(),
            }
            (staging / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # 只保留最近一次导出，避免临时下载文件无限累积。
            for old in list(self.exports_root.iterdir()):
                shutil.rmtree(old, ignore_errors=True)
            target = self.exports_root / export_id
            shutil.move(str(staging), str(target))
            return manifest
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def latest_export(self) -> dict | None:
        latest: dict | None = None
        latest_key = ""
        for d in self.exports_root.iterdir():
            mf = d / "manifest.json"
            if not d.is_dir() or not mf.is_file():
                continue
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            key = f"{m.get('created_at', '')}-{m.get('export_id', '')}"
            if latest is None or key > latest_key:
                latest, latest_key = m, key
        return latest

    def export_file(self, export_id: str, rank: int) -> tuple[Path, dict] | None:
        if (not export_id or export_id.startswith(".")
                or export_id != Path(export_id).name):
            return None
        mf = self.exports_root / export_id / "manifest.json"
        if not mf.is_file():
            return None
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not 0 <= rank < int(manifest.get("ranks", 0)):
            return None
        path = self.exports_root / export_id / f"rank_{rank}.npz"
        return (path, manifest) if path.is_file() else None

    def delete_exports_for(self, archive_id: str) -> None:
        for d in list(self.exports_root.iterdir()):
            mf = d / "manifest.json"
            if not d.is_dir() or not mf.is_file():
                continue
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if m.get("source_archive_id") == archive_id:
                shutil.rmtree(d, ignore_errors=True)
