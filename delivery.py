"""Generate audio; preserve automatic text until the voice send succeeds."""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context
import astrbot.api.message_components as Comp

from .audio_cache import AudioCache
from .configuration import Settings, EffectiveSettings
from .placeholders import resolve
from .tts_client import TTSClient, TTSError, log_parameter_changes

ORIGINAL_DELAY = 0.5
PENDING_KEY = "local_tts_pending"


@dataclass
class PreparedAudio:
    trace: str
    path: Path
    record: object
    text: str
    append_text: bool
    automatic: bool
    sent: bool = False
    error: str = ""
    observed: bool = False
    finished: bool = False
    restore: object = None
    parameter_changes: str = ""


class Delivery:
    def __init__(self, context: Context, cache: AudioCache) -> None:
        self.context = context
        self.cache = cache
        self.client = TTSClient()
        self._limits: dict[str, asyncio.Semaphore] = {}
        self._active = set()
        self._pending = {}
        self._closed = False

    async def prepare(self, event: AstrMessageEvent, settings: Settings,
                      effective: EffectiveSettings, text: str,
                      overrides: dict[str, str | None], automatic: bool,
                      trace: str) -> PreparedAudio:
        task = asyncio.current_task()
        if self._closed:
            raise TTSError("插件正在停止，请稍后重试。")
        self._active.add(task)
        path = None
        try:
            if effective.provider is None:
                raise ConfigError("尚未选择提供商，请在插件设置中选择并保存。")
            limit = self._limits.get(effective.provider.id)
            if limit is None:
                limit = self._limits[effective.provider.id] = asyncio.Semaphore(effective.provider.concurrency)
            async with limit:
                logger.debug(f"[TTS/生成][{trace}] 开始 | 来源={'自动' if automatic else '命令'} | 字数={len(text)}")
                values = await resolve(settings, effective, text, overrides, self.context, trace)
                params = effective.provider.request.render(values)
                audio, suffix = await self.client.synthesize(effective.provider, params, trace)
                try:
                    path = self.cache.store(event, audio, suffix)
                    record = Comp.Record.fromFileSystem(str(path))
                except OSError:
                    raise TTSError("无法保存音频缓存，请检查插件数据目录权限。") from None
                prepared = PreparedAudio(trace, path, record, text, effective.append_text and automatic, automatic)
                prepared.parameter_changes = log_parameter_changes(effective.provider.request, values)
                if not automatic:
                    self._observe(event, prepared)
                return prepared
        except BaseException:
            if path is not None:
                self.cache.release(path)
            raise
        finally:
            self._active.discard(task)

    def _observe(self, event: AstrMessageEvent, prepared: PreparedAudio) -> None:
        """An event-local observer delegates unchanged sends to the original adapter.

        Framework after_message_sent also runs after swallowed send errors. Observing
        the adapter return avoids reporting a failed Record as successful. No global
        class patch, no scheduler re-entry, and unrelated messages pass through.
        """
        original = event.send
        had_override = "send" in vars(event)
        previous = vars(event).get("send")
        task = asyncio.current_task()

        def restore() -> None:
            if vars(event).get("send") is observed_send:
                if had_override:
                    event.send = previous
                else:
                    delattr(event, "send")
            if task is not None:
                task.remove_done_callback(task_done)

        async def observed_send(message):
            chain = getattr(message, "chain", []) if message is not None else []
            ours = any(isinstance(comp, Comp.Record) and
                       (comp is prepared.record or getattr(comp, "file", None) == getattr(prepared.record, "file", None))
                       for comp in chain)
            if not ours or prepared.observed:
                return await original(message)
            prepared.observed = True
            try:
                result = await original(message)
                prepared.sent = True
                logger.info(f"[TTS/发送][{prepared.trace}] 语音发送成功 | 来源={'自动' if prepared.automatic else '命令'} | 内容={json.dumps(prepared.text, ensure_ascii=False)}{prepared.parameter_changes}")
                return result
            except Exception:
                prepared.error = "语音发送失败，请检查平台适配器和音频格式。"
                logger.warning(f"[TTS/发送][{prepared.trace}] 语音发送失败 | 返回命令错误提示")
                # The manual command yields its error after framework sending.
            finally:
                restore()

        def task_done(_):
            self.finish(event, prepared)

        prepared.restore = restore
        event.send = observed_send
        event.set_extra(PENDING_KEY, prepared)
        self._pending[id(prepared)] = (event, prepared)
        if task is not None:
            task.add_done_callback(task_done)

    def finish(self, event: AstrMessageEvent, prepared: PreparedAudio) -> None:
        if prepared.finished:
            return
        prepared.finished = True
        if prepared.restore:
            prepared.restore()
        if event.get_extra(PENDING_KEY) is prepared:
            event.set_extra(PENDING_KEY, None)
        self._pending.pop(id(prepared), None)
        try:
            self.cache.release(prepared.path)
        except Exception:
            logger.warning(f"[TTS/缓存][{prepared.trace}] 释放或清理失败，已跳过")

    async def send_automatic(self, event: AstrMessageEvent, prepared: PreparedAudio, result: MessageEventResult) -> None:
        """Try voice before framework decoration; never reconstruct or resend text.

        On failure, the original result (including metadata) remains untouched.
        Only successful voice delivery may suppress its original text chain.
        """
        task = asyncio.current_task()
        self._active.add(task)
        try:
            if self._closed:
                raise TTSError("插件正在停止，请稍后重试。")
            try:
                await event.send(event.chain_result([prepared.record]))
            except Exception:
                raise TTSError("语音发送失败，原回复继续交由主框架处理。") from None
            prepared.sent = True
            logger.info(f"[TTS/发送][{prepared.trace}] 语音发送成功 | 来源=自动 | 内容={json.dumps(prepared.text, ensure_ascii=False)}{prepared.parameter_changes}")
            if prepared.append_text:
                await asyncio.sleep(ORIGINAL_DELAY)
                logger.debug(f"[TTS/发送][{prepared.trace}] 原回复交由主框架处理 | 间隔≥0.5s")
            else:
                result.chain.clear()
        finally:
            self._active.discard(task)
            self.finish(event, prepared)

    async def close(self) -> None:
        self._closed = True
        tasks = [task for task in self._active if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for event, prepared in list(self._pending.values()):
            # Do not release a file while an adapter is still reading it. Cache.close
            # only removes protection, without deleting; task finalization releases it.
            if prepared.restore:
                prepared.restore()
        self._pending.clear()
        await self.client.close()
