"""Flask 入口：检查点重分片网页。

路由：
  GET  /                 状态页（旧卷 / 新卷 / 发布结果 / 归档 / 历史）
  POST /mount            模拟卸载、挂回同一卷、挂载不同卷
  POST /reshard          选检查点 + 新 rank 数 + 新卷容量 -> 重分片并发布
  GET  /download/<rank>  按新 rank 下载已发布的 npz 分片
  POST /archive/capacity 设置归档容量（字节）
  POST /archive          把当前不依赖旧卷的发布存入回退位（两个位）
  POST /archive/delete   删除任一留档（立即释放独占数组数据）
  POST /archive/export   选留档 + 新 rank 数，仅从该留档重分片并生成下载
  GET  /archive/download/<export_id>/<rank>  下载留档重分片结果
"""
from __future__ import annotations

import os
from pathlib import Path

from flask import (Flask, abort, flash, redirect, render_template, request,
                   send_file)

from archive_store import ArchiveError
from sample_data import ensure_samples
from storage import ReshardService

DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
OTHER_VOLUME_ID = "oldvol-different-000"

service = ReshardService(DATA_DIR)
ensure_samples(service.old)


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = os.environ.get("SECRET_KEY", "reshard-demo-secret")

    @app.get("/")
    def index():
        pub = service.new.published
        readable = service.published_readable()
        return render_template(
            "index.html",
            old=service.old.status(),
            checkpoints=service.old.list_checkpoints(),
            new=service.new,
            published=pub,
            readable=readable,
            usage=service.new.usage_bytes(),
            history=list(reversed(service.new.history)),
            archive=service.archive.page_stats(),
            archive_history=list(reversed(service.archive.history)),
            latest_export=service.archive.latest_export(),
        )

    @app.post("/mount")
    def mount():
        action = request.form.get("action")
        if action == "unmount":
            service.old.unmount()
            flash("旧卷已卸载。依赖旧卷的发布结果现在不可读。", "warn")
        elif action == "mount_same":
            service.old.mount()
            flash(f"已挂回同一卷 {service.old.stored_id()}，依赖结果恢复可读。", "ok")
        elif action == "mount_other":
            service.old.mount(OTHER_VOLUME_ID)
            flash(f"已挂载另一块卷 {OTHER_VOLUME_ID}：卷 ID 不同，"
                  "旧卷内容与依赖它的结果仍不可读。", "warn")
        else:
            flash("未知挂载操作。", "err")
        return redirect("/")

    @app.post("/reshard")
    def reshard():
        ckpt_id = request.form.get("ckpt_id", "")
        try:
            n_new = int(request.form.get("n_new", ""))
            capacity = int(request.form.get("capacity", ""))
        except ValueError:
            flash("新 rank 数和新卷容量必须是整数。", "err")
            return redirect("/")
        if n_new < 1 or capacity < 0:
            flash("新 rank 数必须 >= 1，容量必须 >= 0。", "err")
            return redirect("/")
        result = service.run(
            ckpt_id=ckpt_id,
            n_new=n_new,
            capacity=capacity,
            can_mount_old=request.form.get("can_mount_old") == "on",
            simulate_write_fail=request.form.get("simulate_write_fail") == "on",
        )
        if result["status"] == "published":
            flash(
                f"发布成功 {result['publish_id']}：{n_new} 个新分片，"
                f"训练步 {result['step']}，本次峰值 {result['op_peak']} B"
                f"（上一套 {result['prev_published_bytes']} B + "
                f"本次暂存 {result['staged_bytes']} B）。",
                "ok",
            )
        else:
            flash(f"重分片失败：{result['reason']}。已保留上次发布结果。", "err")
        return redirect("/")

    @app.get("/download/<int:rank>")
    def download(rank: int):
        pub = service.new.published
        if pub is None:
            abort(404, description="尚未发布任何结果")
        if service.published_readable() is False:
            abort(409, description="结果依赖旧卷，旧卷未挂载（或挂载了不同的卷），当前不可读")
        if not 0 <= rank < int(pub["ranks"]):
            abort(404, description=f"rank {rank} 超出范围")
        path = service.new.published_root / pub["id"] / f"rank_{rank}.npz"
        if not path.is_file():
            abort(404, description="分片文件缺失")
        return send_file(path, as_attachment=True,
                         download_name=f"{pub['id']}_rank_{rank}.npz")

    # ---- 归档容量 ----
    @app.post("/archive/capacity")
    def archive_capacity():
        try:
            capacity = int(request.form.get("archive_capacity", ""))
        except ValueError:
            flash("归档容量必须是整数。", "err")
            return redirect("/")
        try:
            service.set_archive_capacity(capacity)
        except ArchiveError as exc:
            flash(f"归档容量未修改：{exc}", "err")
        else:
            flash(f"归档容量已设为 {capacity} B（已保留的数据不删）。", "ok")
        return redirect("/")

    # ---- 存留档 ----
    @app.post("/archive")
    def archive_create():
        try:
            desc = service.archive_current(
                simulate_write_fail=request.form.get("simulate_write_fail") == "on",
            )
        except ArchiveError as exc:
            flash(f"存档失败：{exc}。已有留档与占用保持不变。", "err")
        else:
            flash(
                f"已存档 {desc['id']}：训练步 {desc['step']}，"
                f"新增数组数据 {desc['new_data_bytes']} B，"
                f"数组数据占用 {desc['store_bytes']} B"
                f"（{service.archive.page_stats()['slots_used']}/2 位）。",
                "ok",
            )
        return redirect("/")

    # ---- 删留档 ----
    @app.post("/archive/delete")
    def archive_delete():
        archive_id = request.form.get("archive_id", "")
        try:
            desc = service.delete_archive(archive_id)
        except ArchiveError as exc:
            flash(f"删除失败：{exc}。已有留档与占用保持不变。", "err")
        else:
            extra = "" if desc["pruned"] else (
                "（有留档索引异常，已跳过清理以免误删共享数据；刷新后重试）")
            flash(
                f"已删除留档 {desc['id']}：释放仅该档使用的数组数据 "
                f"{desc['freed_exclusive_bytes']} B，数组数据占用 {desc['store_bytes']} B。"
                + extra,
                "ok",
            )
        return redirect("/")

    # ---- 从留档重分片导出 ----
    @app.post("/archive/export")
    def archive_export():
        archive_id = request.form.get("archive_id", "")
        try:
            n_new = int(request.form.get("n_new", ""))
        except ValueError:
            flash("新 rank 数必须是整数。", "err")
            return redirect("/")
        try:
            manifest = service.export_from_archive(archive_id, n_new)
        except ArchiveError as exc:
            flash(f"从留档重分片失败：{exc}", "err")
        else:
            flash(
                f"已从留档 {manifest['source_archive_id']} 重分片："
                f"{manifest['ranks']} 个分片，训练步 {manifest['step']}，"
                f"下载文件共 {manifest['bytes']} B（不读旧卷、不读当前发布）。",
                "ok",
            )
        return redirect("/")

    @app.get("/archive/download/<export_id>/<int:rank>")
    def archive_download(export_id: str, rank: int):
        found = service.archive.export_file(export_id, rank)
        if found is None:
            abort(404, description="导出结果不存在或 rank 超出范围")
        path, manifest = found
        return send_file(path, as_attachment=True,
                         download_name=f"{manifest['export_id']}_rank_{rank}.npz")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
