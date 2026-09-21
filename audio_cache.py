"""Audio-file retention; never follow links or remove an in-flight file."""
import asyncio
import math
import re
import stat
import time
from pathlib import Path
from typing import Tuple

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent


class AudioCache:
    def __init__(self, directory: Path, config: dict) -> None:
        self.directory = directory
        self._cache_in_use = {}
        self._terminated = False
        self.auto_clean_task = None
        self._warning_times = {}
        self._deferred_at = -float("inf")
        self.cache_limits = {}
        self.cache_config_valid = isinstance(config, dict)
        for key, scale in (("max_age_hours", 3600), ("max_size_mb", 1024 * 1024), ("max_files", 1)):
            try:
                self.cache_limits[key] = self._parse_cache_limit(config.get(key), key, scale)
            except (AttributeError, ValueError, TypeError, OverflowError):
                self.cache_config_valid = False
                self._warn(f"TTS 缓存配置 {key} 无效，自动清理暂停；请留空或填写正数。")
        self.cache_auto_enabled = self.cache_config_valid and any(
            v is not None for v in self.cache_limits.values()
        )

    def _warn(self, message: str) -> None:
        # Repeated I/O failures are summarized at most once per five minutes.
        now = time.monotonic()
        if now - self._warning_times.get(message, -float("inf")) >= 300:
            self._warning_times[message] = now
            logger.warning("[TTS/缓存] " + message)

    def start(self) -> None:
        if self.cache_auto_enabled:
            self.auto_clean_task = asyncio.create_task(self._auto_clean_task())

    def store(self, event: AstrMessageEvent, content: bytes, suffix: str) -> Path:
        from uuid import uuid4
        if not self._cache_directory_safe():
            raise OSError("Unsafe cache directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"tts_{uuid4().hex}{suffix}"
        self._protect_audio(event, path)
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except OSError:
            self._release_audio(path)
            raise
        return path

    def release(self, path: Path) -> None:
        self._release_audio(path)
        self._run_cache_cleanup("发送后")

    def clean(self) -> tuple[int, float, int]:
        return self._clean_cache(force=True, trigger="手动命令")

    async def close(self) -> None:
        self._terminated = True
        if self.auto_clean_task:
            self.auto_clean_task.cancel()
            try:
                await self.auto_clean_task
            except asyncio.CancelledError:
                pass
        for path in list(self._cache_in_use):
            self._release_audio(path)

    @staticmethod
    def _parse_cache_limit(value, key: str, scale: int):
        """空值不限制；返回秒、字节或文件数，不把非法输入解释成零。"""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        text = str(value).strip()
        if key == "max_files":
            if not re.fullmatch(r"[0-9]+", text):
                raise ValueError(key)
            number = int(text)
        else:
            try:
                number = float(text) * scale
            except (ValueError, OverflowError):
                raise ValueError(key) from None
            if not math.isfinite(number):
                raise ValueError(key)
        if number <= 0:
            raise ValueError(key)
        return number

    @staticmethod
    def _is_link(file_stat) -> bool:
        # Windows junction 等 reparse point 也不能跟随。
        return stat.S_ISLNK(file_stat.st_mode) or bool(
            getattr(file_stat, "st_file_attributes", 0) & 0x400
        )

    def _cache_directory_safe(self) -> bool:
        """拒绝缓存目录及其父目录上的符号链接/目录联接。"""
        directory = self.directory.absolute()
        try:
            for path in (directory, *directory.parents):
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if self._is_link(info) or not stat.S_ISDIR(info.st_mode):
                    self._warn("缓存目录或其父目录不是普通目录，已跳过缓存操作。")
                    return False
        except OSError:
            self._warn("无法检查缓存目录，已跳过缓存操作。")
            return False
        return True

    def _clean_cache(self, force: bool = False, trigger: str = "检查") -> Tuple[int, float, int]:
        """按约束淘汰；force 手动清空。返回成功删除数、MB、在用跳过数。

        扫描、写入和保护登记均在事件循环中同步执行且不 await，避免清理相互穿插。
        """
        if not force and not self.cache_auto_enabled:
            return 0, 0.0, 0
        if not self._cache_directory_safe():
            return 0, 0.0, 0
        files = []
        try:
            for path in self.directory.iterdir():
                try:
                    info = path.lstat()
                    if stat.S_ISREG(info.st_mode) and not self._is_link(info):
                        files.append((path, info))
                except FileNotFoundError:
                    continue
                except OSError:
                    # 不能完整统计时，不据此做破坏性清理。
                    self._warn("无法读取缓存文件信息，本轮清理已跳过。")
                    return 0, 0.0, 0
        except FileNotFoundError:
            if force:
                logger.info("[TTS/缓存] 清理完成 | 原因=手动命令 | 删除=0个/0.00MB")
            return 0, 0.0, 0
        except OSError:
            self._warn("无法扫描缓存目录，本轮清理已跳过。")
            return 0, 0.0, 0

        files.sort(key=lambda entry: (entry[1].st_mtime, entry[0].name))
        remaining_count = len(files)
        remaining_bytes = sum(info.st_size for _, info in files)
        max_age = self.cache_limits.get("max_age_hours")
        max_size = self.cache_limits.get("max_size_mb")
        max_count = self.cache_limits.get("max_files")
        now = time.time()
        count = removed_bytes = skipped = 0
        reasons = []
        if force:
            reasons.append("手动命令")
        else:
            if max_age is not None and any(now - info.st_mtime >= max_age for _, info in files):
                reasons.append(f"超过保留时间({max_age / 3600:g}h)")
            if max_size is not None and remaining_bytes > max_size:
                reasons.append(f"容量超限({max_size / (1024 * 1024):g}MB)")
            if max_count is not None and remaining_count > max_count:
                reasons.append(f"数量超限({max_count}个)")

        def over_capacity():
            return ((max_size is not None and remaining_bytes > max_size)
                    or (max_count is not None and remaining_count > max_count))

        expired_remaining = False
        for path, info in files:
            expired = max_age is not None and now - info.st_mtime >= max_age
            if path in self._cache_in_use:
                skipped += 1
                expired_remaining |= expired
                continue
            if not (force or expired or over_capacity()):
                continue
            try:
                # 防止扫描后文件被外部进程替换；不删除新版本或链接。
                current = path.lstat()
                if (self._is_link(current) or not stat.S_ISREG(current.st_mode)
                        or (current.st_ino, current.st_size, current.st_mtime_ns)
                        != (info.st_ino, info.st_size, info.st_mtime_ns)):
                    expired_remaining |= expired
                    continue
                path.unlink()
            except FileNotFoundError:
                remaining_count -= 1
                remaining_bytes -= info.st_size
                continue
            except OSError:
                expired_remaining |= expired
                self._warn("存在无法删除的缓存文件，将在后续清理时重试。")
                continue
            count += 1
            removed_bytes += info.st_size
            remaining_count -= 1
            remaining_bytes -= info.st_size

        if reasons:
            deferred = not force and (over_capacity() or expired_remaining)
            now_mono = time.monotonic()
            if count or force or now_mono - self._deferred_at >= 300:
                status = "部分暂缓" if deferred else "清理完成"
                logger.info(f"[TTS/缓存] {status} | 检查={trigger} | 原因={'、'.join(reasons)} | 删除={count}个/{removed_bytes / (1024 * 1024):.2f}MB | 在用跳过={skipped}个")
                self._deferred_at = now_mono
        return count, removed_bytes / (1024 * 1024), skipped

    def _run_cache_cleanup(self, trigger: str = "检查") -> None:
        if not self._terminated:
            self._clean_cache(trigger=trigger)

    def _protect_audio(self, event: AstrMessageEvent, path: Path) -> None:
        task = asyncio.current_task()
        # 正常发送在 finally 中释放；任务异常结束时提供第二道释放保障。
        # 不用定时器猜测文件是否仍在使用。
        def task_done(_):
            self._release_audio(path)
            self._run_cache_cleanup("发送后")
        self._cache_in_use[path] = (event, task, task_done)
        if task is not None:
            task.add_done_callback(task_done)

    def _release_audio(self, path: Path) -> None:
        entry = self._cache_in_use.pop(path, None)
        if entry is not None:
            _, task, callback = entry
            if task is not None:
                task.remove_done_callback(callback)

    async def _auto_clean_task(self) -> None:
        """启动时检查一次，之后每分钟检查；三项全空时不创建本任务。"""
        trigger = "启动"
        try:
            while True:
                try:
                    self._run_cache_cleanup(trigger)
                except Exception:
                    self._warn("TTS缓存自动清理任务出现错误。")
                trigger = "定时"
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

