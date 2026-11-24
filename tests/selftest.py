"""交付前自测：重分片逻辑 + 换卡 + 卸载/挂回 + 失败保留旧发布 + 归档回退位。

运行: python3 tests/selftest.py
退出码 0 = 全部通过。
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import reshard  # noqa: E402
from archive_store import (ArchiveCapacityError, ArchiveSlotFullError,  # noqa: E402
                           ArchiveStore, ArchiveWriteError, CHUNK_SIZE)
from reshard import (ARRAY_KEYS, MissingShardError, OverCapacityError,  # noqa: E402
                     SimulatedWriteError, StepMismatchError, split_sizes)
from sample_data import ensure_samples  # noqa: E402
from storage import ReshardService  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"[PASS] {name}")
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc(limit=3)
        results.append((FAIL, name, tb))
        print(f"[FAIL] {name}\n{tb}")


def make_service(tmp: str) -> ReshardService:
    svc = ReshardService(Path(tmp))
    ensure_samples(svc.old)
    svc.new.set_capacity(1 << 20)
    svc.set_archive_capacity(1 << 20)
    return svc


def load_source_arrays(data_dir: Path, ckpt_id: str):
    """直接从旧卷目录读源数组，用于和发布结果比对。"""
    ckpt = data_dir / "old_volume" / "checkpoints" / ckpt_id
    per_rank = {k: [] for k in ARRAY_KEYS}
    r = 0
    while (ckpt / f"rank_{r}.npz").is_file():
        with np.load(ckpt / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                per_rank[k].append(z[k])
        r += 1
    return {k: np.concatenate(v) for k, v in per_rank.items()}


def published_arrays(svc: ReshardService):
    pub = svc.new.published
    out = {k: [] for k in ARRAY_KEYS}
    steps = []
    for r in range(pub["ranks"]):
        with np.load(svc.new.published_root / pub["id"] / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                out[k].append(z[k])
            steps.append(int(z["step"]))
    return {k: np.concatenate(v) for k, v in out.items()}, steps


# ---------- 1. 纯逻辑：等分规则 ----------
def t_split_sizes():
    assert split_sizes(30, 5) == [6, 6, 6, 6, 6]
    assert split_sizes(10, 3) == [4, 3, 3], "余数应给低 rank"
    assert split_sizes(31, 4) == [8, 8, 8, 7]
    assert split_sizes(2, 5) == [1, 1, 0, 0, 0], "rank 多于元素时低 rank 先分"
    assert split_sizes(7, 1) == [7]
    assert sum(split_sizes(100, 9)) == 100


# ---------- 2. 换卡 3 -> 5：拼接一致 + 每片带训练步 ----------
def t_reshard_3_to_5():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 5, 1 << 20, True, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [6, 6, 6, 6, 6]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, steps = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k]), f"{k} 拼接后与源不一致"
        assert steps == [42000] * 5, "每片都应带训练步"
        assert svc.new.published["step"] == 42000


# ---------- 3. 换卡 3 -> 2：余数给低 rank ----------
def t_reshard_3_to_2():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [15, 15]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        got, _ = published_arrays(svc)
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k])


# ---------- 4. 奇数总量余数分布 ----------
def t_remainder_to_low_ranks():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 7, 1 << 20, False, False)  # 30 = 7*4+2
        assert op["status"] == "published", op
        assert op["shard_sizes"] == [5, 5, 4, 4, 4, 4, 4], op["shard_sizes"]


# ---------- 5. 缺片：失败并保留上次发布 ----------
def t_missing_shard_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        ok = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert ok["status"] == "published"
        before = svc.new.published["id"]
        bad = svc.run("ckpt-missing-rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "缺片" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before, "失败后应保留上次发布"
        assert svc.new.staging_bytes() == 0, "失败后暂存应清理"


# ---------- 6. 训练步不一致：失败并保留上次发布 ----------
def t_step_mismatch_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-step-mismatch", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "训练步不一致" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before


# ---------- 7. 容量超限（上一套 + 本次暂存都计入） ----------
def t_over_capacity_keeps_published():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        prev_bytes = svc.new.published_bytes()
        # 容量只比上一套大一点，必然装不下上一套 + 本次暂存
        bad = svc.run("ckpt-demo-3rank", 4, prev_bytes + 10, False, False)
        assert bad["status"] == "failed"
        assert "容量超限" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 8. 模拟写失败：保留上次发布，暂存清理 ----------
def t_simulated_write_failure():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        before = svc.new.published["id"]
        bad = svc.run("ckpt-demo-3rank", 6, 1 << 20, False, True)
        assert bad["status"] == "failed"
        assert "模拟写失败" in bad["reason"], bad["reason"]
        assert svc.new.published["id"] == before
        assert svc.new.staging_bytes() == 0


# ---------- 9. 峰值 = 上一套 + 本次暂存 ----------
def t_peak_accounting():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op1 = svc.run("ckpt-demo-3rank", 2, 1 << 20, False, False)
        assert op1["op_peak"] == 0 + op1["staged_bytes"]
        peak1 = svc.new.peak_bytes
        op2 = svc.run("ckpt-demo-3rank", 5, 1 << 20, False, False)
        assert op2["op_peak"] == op2["prev_published_bytes"] + op2["staged_bytes"]
        assert op2["prev_published_bytes"] > 0, "第二套发布时上一套应计入"
        assert svc.new.peak_bytes == max(peak1, op2["op_peak"])


# ---------- 10. 卸载 -> 依赖旧卷的结果不可读；挂回同一卷恢复；挂不同卷不恢复 ----------
def t_unmount_dependency_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        op = svc.run("ckpt-demo-3rank", 4, 1 << 20, True, False)  # 依赖旧卷
        assert op["status"] == "published"
        assert svc.published_readable() is True
        svc.old.unmount()
        assert svc.published_readable() is False, "卸载后应不可读"
        svc.old.mount("oldvol-different-000")
        assert svc.published_readable() is False, "挂不同卷不应恢复"
        svc.old.mount()  # 挂回同一卷
        assert svc.published_readable() is True, "挂回同一卷应恢复"


# ---------- 11. 不依赖旧卷的结果：卸载后仍可读 ----------
def t_independent_result_survives_unmount():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)  # 不依赖旧卷
        svc.old.unmount()
        assert svc.published_readable() is True


# ---------- 12. 卸载时源不可读，不能重分片 ----------
def t_unmounted_source_unreadable():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        svc.old.unmount()
        bad = svc.run("ckpt-demo-3rank", 4, 1 << 20, False, False)
        assert bad["status"] == "failed"
        assert "源检查点不可读" in bad["reason"], bad["reason"]
        assert svc.new.published is None


# ---------- 13. HTTP 冒烟：页面 / 重分片 / 下载 / 卸载后 409 / 挂回恢复 ----------
def t_http_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        # 延迟导入，让 server 用临时 DATA_DIR
        for mod in ("server",):
            sys.modules.pop(mod, None)
        import server  # noqa: WPS433

        c = server.app.test_client()
        assert c.get("/").status_code == 200

        r = c.post("/reshard", data={
            "ckpt_id": "ckpt-demo-3rank", "n_new": "5",
            "capacity": str(1 << 20), "can_mount_old": "on",
        }, follow_redirects=False)
        assert r.status_code == 302
        pub = server.service.new.published
        assert pub and pub["ranks"] == 5

        # 下载 rank_3 并核对内容 = 全局下标 [18, 24)
        resp = c.get("/download/3")
        assert resp.status_code == 200
        data = np.load(io.BytesIO(resp.data))
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        for k in ARRAY_KEYS:
            assert np.array_equal(data[k], src[k][18:24]), f"下载分片 {k} 内容错误"
        assert int(data["step"]) == 42000

        # 卸载 -> 依赖旧卷，下载应 409
        c.post("/mount", data={"action": "unmount"})
        assert c.get("/download/3").status_code == 409
        page = c.get("/").data.decode()
        assert "不可读" in page

        # 挂回同一卷 -> 恢复
        c.post("/mount", data={"action": "mount_same"})
        assert c.get("/download/3").status_code == 200


# =====================================================================
# 归档回退位
# =====================================================================
def archive_bytes(svc: ReshardService, archive_id: str) -> int:
    return svc.archive.exclusive_bytes(archive_id)


def read_published_global(svc: ReshardService) -> dict:
    """读当前发布拼出的全局数组（不经过归档）。"""
    pub = svc.new.published
    per = {k: [] for k in ARRAY_KEYS}
    for r in range(pub["ranks"]):
        with np.load(svc.new.published_root / pub["id"] / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                per[k].append(z[k])
    return {k: np.concatenate(v) for k, v in per.items()}


def archive_download_arrays(svc: ReshardService, archive_id: str, n_new: int):
    m = svc.export_from_archive(archive_id, n_new)
    assert m["ranks"] == n_new
    got = {k: [] for k in ARRAY_KEYS}
    steps = []
    for r in range(n_new):
        with np.load(svc.archive.exports_root / m["export_id"] / f"rank_{r}.npz") as z:
            for k in ARRAY_KEYS:
                got[k].append(z[k])
            steps.append(int(z["step"]))
    return m, {k: np.concatenate(v) for k, v in got.items()}, steps


def publish(svc: ReshardService, ckpt="ckpt-demo-3rank", n=2, **kw):
    op = svc.run(ckpt, n, kw.pop("capacity", 1 << 20),
                 kw.pop("can_mount_old", False), kw.pop("simulate_write_fail", False))
    assert op["status"] == "published", op
    return op


# ---------- 14. 两个回退位：第三份被拒，已有两档不变 ----------
def t_archive_two_slots():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc)
        a1 = svc.archive_current()
        publish(svc, "ckpt-demo-3rank-step2")
        a2 = svc.archive_current()
        assert svc.archive.page_stats()["slots_used"] == 2
        before = svc.archive.stored_bytes()
        publish(svc, n=5)
        try:
            svc.archive_current()
        except ArchiveSlotFullError as exc:
            assert "两个回退位都已占用" in str(exc), exc
        else:
            raise AssertionError("第三份存档应被拒绝")
        assert svc.archive.stored_bytes() == before, "被拒的存档不应改变占用"
        assert [a["id"] for a in svc.archive.archives] == [a1["id"], a2["id"]]

        # 删一个后可再存
        svc.delete_archive(a1["id"])
        a3 = svc.archive_current()
        assert len(svc.archive.archives) == 2
        assert {a["id"] for a in svc.archive.archives} == {a2["id"], a3["id"]}


# ---------- 15. 仅训练步不同：数组数据零新增 ----------
def t_archive_step_only_zero_growth():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc)
        a1 = svc.archive_current()
        assert a1["new_data_bytes"] == svc.archive.stored_bytes()
        publish(svc, "ckpt-demo-3rank-step2")  # 同数组，仅 step 43000
        a2 = svc.archive_current()
        assert a2["new_data_bytes"] == 0, a2
        assert a2["step"] == 43000 and a1["step"] == 42000
        assert svc.archive.stored_bytes() == a1["new_data_bytes"], "共享数组数据不应重复占用"
        for a in svc.archive.archives:
            assert archive_bytes(svc, a["id"]) == 0, "两档互相共享，独占应为 0"
        # 两档都能读回各自训练步
        for arch, step in ((a1, 42000), (a2, 43000)):
            loaded = svc.archive.load_arrays(arch["id"])
            assert loaded["step"] == step
        # 归档历史记录成功
        hist = [h for h in svc.archive.history if h["kind"] == "archive_create"]
        assert len(hist) == 2 and all(h["status"] == "archived" for h in hist)


# ---------- 16. 仅 rank 切法不同：数组数据零新增 ----------
def t_archive_rank_only_zero_growth():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc, n=2)
        a1 = svc.archive_current()
        publish(svc, n=7)  # 同一源，只换 rank 切法
        a2 = svc.archive_current()
        assert a2["new_data_bytes"] == 0, a2
        assert svc.archive.stored_bytes() == a1["new_data_bytes"]
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        for arch in (a1, a2):
            loaded = svc.archive.load_arrays(arch["id"])
            for k in ARRAY_KEYS:
                assert np.array_equal(loaded["arrays"][k], src[k])


# ---------- 17. 同 dtype 同长度，仅一段连续变化 -> 新增 <= 改动段 + 8 KiB ----------
def _make_demo_variant(base: Path, ckpt_id: str, step: int, mutator) -> None:
    """基于 demo 全局数组造一个变体检查点（同 rank 切法、同 dtype）。"""
    from sample_data import DEMO_LENGTHS, _demo_arrays, _write_rank, _manifest
    ckpt = base / ckpt_id
    ckpt.mkdir(parents=True, exist_ok=False)
    arrays = _demo_arrays()
    for k in ARRAY_KEYS:
        mutator(arrays[k])
    off = 0
    for r, ln in enumerate(DEMO_LENGTHS):
        sl = slice(off, off + ln)
        _write_rank(ckpt, r, *(arrays[k][sl] for k in ARRAY_KEYS), step=step)
        off += ln
    _manifest(ckpt, ckpt_id, len(DEMO_LENGTHS), step, DEMO_LENGTHS, "float32")


def t_archive_small_change_bound():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        src_ckpt = svc.old.checkpoints_dir()
        # 基础发布：为了超过一个块，额外造一个 64 KiB 数组的检查点（同 dtype）
        big_id = "ckpt-big-64k"
        big_dir = src_ckpt / big_id
        big_dir.mkdir()
        rng = np.random.default_rng(99)
        n = 64 * 1024 // 4  # 16384 个 float32 -> 每数组 64 KiB
        base_arrays = {k: rng.standard_normal(n).astype(np.float32) for k in ARRAY_KEYS}
        # 保证可复现的变体：改动中间一段连续数据
        from sample_data import _write_rank, _manifest
        _write_rank(big_dir, 0, *(base_arrays[k] for k in ARRAY_KEYS), step=100)
        _manifest(big_dir, big_id, 1, 100, [n], "float32")
        publish(svc, big_id, n=1)
        a1 = svc.archive_current()

        # 变体：每个数组都在同一处连续改 1500 字节（不足一块，但会跨 2 块）
        change_bytes = 1500  # 非 4096 对齐，最多弄脏 ceil(1500/4096)+1 = 2 块/数组
        variant_arrays = {k: base_arrays[k].copy() for k in ARRAY_KEYS}
        start_byte = 20000  # 非块对齐
        elems = change_bytes // 4
        e0 = start_byte // 4
        for k in ARRAY_KEYS:
            variant_arrays[k][e0:e0 + elems] += np.float32(1.0)
        big_id2 = "ckpt-big-64k-v1"
        vdir = src_ckpt / big_id2
        vdir.mkdir()
        _write_rank(vdir, 0, *(variant_arrays[k] for k in ARRAY_KEYS), step=101)
        _manifest(vdir, big_id2, 1, 101, [n], "float32")

        publish(svc, big_id2, n=1)
        a2 = svc.archive_current()
        per_array = a2["new_data_bytes_by_array"]
        limit = change_bytes + 8 * 1024
        for k in ARRAY_KEYS:
            assert per_array[k] <= limit, (k, per_array[k], limit)
            assert per_array[k] > 0, f"{k} 确实改动了，应有新增"
        # 同一档内三个数组各自独立受这条上限约束
        assert all(per_array[k] <= limit for k in ARRAY_KEYS)
        # 未改动的部分逐块共享：新增总量也远小于 3 份完整数组
        assert a2["new_data_bytes"] < 3 * 64 * 1024
        print(f"      实测：改动段 {change_bytes} B/数组，新增 "
              f"{per_array} B（上限 {limit} B/数组）")
        # 保留第一档不动，第二档读回内容正确
        loaded = svc.archive.load_arrays(a2["id"])
        for k in ARRAY_KEYS:
            assert np.array_equal(loaded["arrays"][k], variant_arrays[k])
        loaded1 = svc.archive.load_arrays(a1["id"])
        for k in ARRAY_KEYS:
            assert np.array_equal(loaded1["arrays"][k], base_arrays[k])


# ---------- 18. 占用 = 去重后数据块文件实际字节 ----------
def t_archive_usage_equals_files():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc)
        svc.archive_current()
        publish(svc, n=9)
        svc.archive_current()
        store = svc.archive.store_root
        files = [p for p in store.rglob("*") if p.is_file()]
        actual = sum(p.stat().st_size for p in files)
        assert svc.archive.stored_bytes() == actual
        # 块按 4 KiB 上限，没有索引/清单算进数组占用
        assert all(p.stat().st_size <= CHUNK_SIZE for p in files)
        # 数组数据只被共享，两档独占均为 0
        assert all(a["exclusive_bytes"] == 0 for a in svc.archive.page_stats()["archives"])
        # 逻辑字节数是各档 3 个数组的原始字节之和（两档）
        stats = svc.archive.page_stats()
        pub = svc.new.published
        assert stats["logical_bytes"] == pub["total_elems"] * 4 * 3 * 2


# ---------- 19. 归档超额：已有留档与占用照旧 ----------
def t_archive_over_capacity_keeps():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        big_id = "ckpt-big-64k"
        big_dir = svc.old.checkpoints_dir() / big_id
        big_dir.mkdir()
        rng = np.random.default_rng(5)
        n = 4096
        arrays = {k: rng.standard_normal(n).astype(np.float32) for k in ARRAY_KEYS}
        from sample_data import _write_rank, _manifest
        _write_rank(big_dir, 0, *(arrays[k] for k in ARRAY_KEYS), step=1)
        _manifest(big_dir, big_id, 1, 1, [n], "float32")
        svc.set_archive_capacity(1 << 20)
        publish(svc, big_id, n=1)
        a1 = svc.archive_current()
        before_used = svc.archive.stored_bytes()
        assert a1["new_data_bytes"] > 0

        # 新数据与 a1 完全不同，容量只够 a1
        publish(svc, n=3)
        svc.set_archive_capacity(before_used + 16)
        try:
            svc.archive_current()
        except ArchiveCapacityError as exc:
            assert "归档容量超限" in str(exc), exc
        else:
            raise AssertionError("超额应失败")
        assert svc.archive.stored_bytes() == before_used, "超额不应改变占用"
        assert len(svc.archive.archives) == 1
        assert svc.archive.capacity == before_used + 16, "容量设置本身保留"
        # 已保留留档仍可完整读回
        loaded = svc.archive.load_arrays(a1["id"])
        for k in ARRAY_KEYS:
            assert np.array_equal(loaded["arrays"][k], arrays[k])


# ---------- 20. 归档写失败：已有留档与占用照旧 ----------
def t_archive_write_fail_keeps():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc)
        a1 = svc.archive_current()
        before_used = svc.archive.stored_bytes()
        before_ids = [a["id"] for a in svc.archive.archives]
        # 换一版数据（每个数组都变化），确保本次确实要写新块
        _make_demo_variant(svc.old.checkpoints_dir(), "ckpt-variant-wf", 42001,
                           lambda arr: np.copyto(arr, arr + np.float32(3.0)))
        publish(svc, "ckpt-variant-wf", n=4)
        try:
            svc.archive_current(simulate_write_fail=True)
        except ArchiveWriteError as exc:
            assert "模拟归档写失败" in str(exc), exc
        else:
            raise AssertionError("模拟写失败应抛错")
        assert svc.archive.stored_bytes() == before_used, "写失败应回滚新块"
        assert [a["id"] for a in svc.archive.archives] == before_ids
        assert svc.archive.exclusive_bytes(a1["id"]) == before_used
        assert len([p for p in svc.archive.store_root.rglob("*") if p.is_file()]) == 3, \
            "只应剩留档 a1 的 3 个块（每数组 120 B）"
        loaded = svc.archive.load_arrays(a1["id"])
        assert loaded["step"] == 42000


# ---------- 21. 依赖旧卷的发布不能存档 ----------
def t_archive_rejects_old_dependent():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc, can_mount_old=True)
        try:
            svc.archive_current()
        except Exception as exc:  # ArchiveError
            assert "依赖旧卷" in str(exc), exc
        else:
            raise AssertionError("依赖旧卷的发布不应可存档")
        assert svc.archive.stored_bytes() == 0


# ---------- 22. 删档只释放独占数据，保留档可读且无需重写 ----------
def t_archive_delete_frees_exclusive():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc)
        a1 = svc.archive_current()           # demo 数据
        publish(svc, "ckpt-missing-rank" if False else "ckpt-demo-3rank", n=3)
        # 第二档用不同数据：造一个变体检查点
        _make_demo_variant(svc.old.checkpoints_dir(), "ckpt-variant", 42001,
                           lambda arr: np.copyto(arr, arr + np.float32(2.0)))
        publish(svc, "ckpt-variant", n=3)
        a2 = svc.archive_current()
        used2 = svc.archive.stored_bytes()
        assert a2["new_data_bytes"] > 0
        excl1 = svc.archive.exclusive_bytes(a1["id"])
        excl2 = svc.archive.exclusive_bytes(a2["id"])
        assert excl1 > 0 and excl2 > 0, (excl1, excl2)
        assert excl1 + excl2 == used2, "两档无共享时独占之和 = 占用"

        # 记录保留档的 index 与块文件 mtime/inode，确认删档不重写保留档
        keep_index = svc.archive._index_path(a2["id"])
        keep_bytes = keep_index.read_bytes()
        keep_chunks = set(svc.archive._archive_hashes(a2["id"]))
        keep_stat = {h: (svc.archive._chunk_path(h).stat().st_ino,
                         svc.archive._chunk_path(h).stat().st_mtime_ns)
                     for h in keep_chunks}

        desc = svc.delete_archive(a1["id"])
        assert desc["freed_exclusive_bytes"] == excl1
        assert svc.archive.stored_bytes() == excl2, "删档应立即释放独占数据"
        assert not svc.archive._index_path(a1["id"]).exists()
        assert svc.archive._index_path(a2["id"]).read_bytes() == keep_bytes, "保留档索引被改写"
        for h in keep_chunks:
            p = svc.archive._chunk_path(h)
            assert p.is_file(), f"保留档数据块被误删: {h[:12]}"
            assert (p.stat().st_ino, p.stat().st_mtime_ns) == keep_stat[h], "保留档数据块被重写"
        loaded = svc.archive.load_arrays(a2["id"])
        assert loaded["step"] == 42001
        # 删档后配额空出一位，可再存
        publish(svc, n=2)
        svc.archive_current()
        assert len(svc.archive.archives) == 2


# ---------- 23. 旧卷卸载后，从留档重分片并下载 ----------
def t_archive_export_after_unmount():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc, n=3)
        arch = svc.archive_current()
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        svc.old.unmount()
        assert svc.published_readable() is True  # 存档的是不依赖旧卷的发布
        m, got, steps = archive_download_arrays(svc, arch["id"], 4)
        assert m["ranks"] == 4 and m["shard_sizes"] == [8, 8, 7, 7]
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k]), f"{k} 从留档重分片内容错误"
        assert steps == [42000] * 4
        # 只从留档读：把当前发布整目录移走，导出仍成功
        pub_dir = svc.new.published_root / svc.new.published["id"]
        moved = Path(tmp) / "moved-pub"
        shutil.move(str(pub_dir), str(moved))
        m2, got2, steps2 = archive_download_arrays(svc, arch["id"], 7)
        assert m2["shard_sizes"] == [5, 5, 4, 4, 4, 4, 4]
        for k in ARRAY_KEYS:
            assert np.array_equal(got2[k], src[k])
        assert steps2 == [42000] * 7


# ---------- 24. 当前发布换版后，从留档重分片并下载 ----------
def t_archive_export_after_republish():
    with tempfile.TemporaryDirectory() as tmp:
        svc = make_service(tmp)
        publish(svc, n=2)
        arch = svc.archive_current()
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        old_pub = svc.new.published["id"]
        # 发布换版：不同 rank 切法与不同训练步
        publish(svc, "ckpt-demo-3rank-step2", n=5)
        assert svc.new.published["id"] != old_pub
        m, got, steps = archive_download_arrays(svc, arch["id"], 6)
        assert m["shard_sizes"] == [5, 5, 5, 5, 5, 5]
        for k in ARRAY_KEYS:
            assert np.array_equal(got[k], src[k])
        assert steps == [42000] * 6, "应读留档的训练步，而非当前发布的 43000"


# ---------- 25. 归档 HTTP：容量/存档/删档/导出下载 ----------
def t_archive_http():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        sys.modules.pop("server", None)
        import server  # noqa: WPS433

        c = server.app.test_client()
        assert c.get("/").status_code == 200
        # 未发布时存档按钮禁用，直接 POST 也失败但不崩
        r = c.post("/archive", data={})
        assert r.status_code == 302
        assert server.service.archive.stored_bytes() == 0

        c.post("/reshard", data={"ckpt_id": "ckpt-demo-3rank", "n_new": "3",
                                 "capacity": str(1 << 20)})
        assert server.service.new.published["ranks"] == 3
        c.post("/archive/capacity", data={"archive_capacity": str(1 << 20)})
        c.post("/archive", data={})
        a1 = server.service.archive.archives[0]
        c.post("/reshard", data={"ckpt_id": "ckpt-demo-3rank-step2", "n_new": "5",
                                 "capacity": str(1 << 20)})
        c.post("/archive", data={})
        assert len(server.service.archive.archives) == 2

        # 页面显示入档后实际保留的数组数据字节数
        page = c.get("/").data.decode()
        used = server.service.archive.stored_bytes()
        assert str(used) in page
        assert "数组数据占用" in page

        # 从留档导出并下载
        r = c.post("/archive/export", data={"archive_id": a1["id"], "n_new": "4"})
        assert r.status_code == 302
        exp = server.service.archive.latest_export()
        assert exp and exp["source_archive_id"] == a1["id"] and exp["ranks"] == 4
        resp = c.get(f"/archive/download/{exp['export_id']}/2")
        assert resp.status_code == 200
        data = np.load(io.BytesIO(resp.data))
        src = load_source_arrays(Path(tmp), "ckpt-demo-3rank")
        # 30 -> 4 片: [8,8,7,7]; rank 2 是全局 [16,23)
        for k in ARRAY_KEYS:
            assert np.array_equal(data[k], src[k][16:23])
        assert int(data["step"]) == 42000

        # 删档
        c.post("/archive/delete", data={"archive_id": a1["id"]})
        assert [a["id"] for a in server.service.archive.archives] != [a1["id"]]
        assert len(server.service.archive.archives) == 1
        # 导出文件也随删档清理
        assert server.service.archive.latest_export() is None
        assert c.get(f"/archive/download/{exp['export_id']}/2").status_code == 404

        # 容量改动在失败/成功都保留
        c.post("/archive/capacity", data={"archive_capacity": "12345"})
        assert server.service.archive.capacity == 12345
        c.post("/archive/capacity", data={"archive_capacity": "abc"})
        assert server.service.archive.capacity == 12345


def main() -> int:
    tests = [
        ("等分规则（余数给低 rank）", t_split_sizes),
        ("换卡 3->5：拼接一致、每片带训练步", t_reshard_3_to_5),
        ("换卡 3->2：连续等分", t_reshard_3_to_2),
        ("余数分布 30->7 片", t_remainder_to_low_ranks),
        ("缺片：失败并保留上次发布", t_missing_shard_keeps_published),
        ("训练步不一致：失败并保留上次发布", t_step_mismatch_keeps_published),
        ("容量超限：失败并保留上次发布", t_over_capacity_keeps_published),
        ("模拟写失败：保留上次发布、清理暂存", t_simulated_write_failure),
        ("峰值 = 上一套 + 本次暂存", t_peak_accounting),
        ("卸载/挂不同卷/挂回同一卷的可读性", t_unmount_dependency_cycle),
        ("不依赖旧卷的结果卸载后仍可读", t_independent_result_survives_unmount),
        ("卸载时源不可读，重分片失败", t_unmounted_source_unreadable),
        ("HTTP 冒烟：页面/发布/下载/409/恢复", t_http_smoke),
        # ---- 归档回退位 ----
        ("归档：留两个位、第三份被拒", t_archive_two_slots),
        ("归档：仅 step 不同 -> 数组零新增", t_archive_step_only_zero_growth),
        ("归档：仅 rank 切法不同 -> 数组零新增", t_archive_rank_only_zero_growth),
        ("归档：连续 8 KiB 变更新增 <= 改动段 + 8 KiB", t_archive_small_change_bound),
        ("归档：库内占用 = 去重后实际字节", t_archive_usage_equals_files),
        ("归档：超额失败保留留档与占用", t_archive_over_capacity_keeps),
        ("归档：写失败保留留档与占用", t_archive_write_fail_keeps),
        ("归档：依赖旧卷的发布不可存档", t_archive_rejects_old_dependent),
        ("归档：删档只释放独占数据、保留档可读", t_archive_delete_frees_exclusive),
        ("归档：卸载旧卷后从留档重分片并下载", t_archive_export_after_unmount),
        ("归档：发布换版后从留档重分片并下载", t_archive_export_after_republish),
        ("归档 HTTP：存档/删档/改容量/导出下载", t_archive_http),
    ]
    for name, fn in tests:
        check(name, fn)
    failed = [r for r in results if r[0] == FAIL]
    print(f"\n{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
