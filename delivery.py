"""Prepare audio; let AstrBot send it and observe only this event's own Record."""
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
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
        self.limit = asyncio.Semaphore(2)
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
            async with self.limit:
                logger.debug(f"[TTS/生成][{trace}] 开始 | 来源={'自动' if automatic else '命令'} | 字数={len(text)}")
                values = await resolve(settings, effective, text, overrides, self.context, trace)
                params = effective.provider.request.render(values)
                audio, suffix = await self.client.synthesize(effective.provider, params, trace)
                try:
                    path = self.cache.store(event, audio, suffix)
                    record = Comp.Record.fromFileSystem(str(path))
                except OSError:
                    raise TTSError("无法保存音频缓存，请检查插件数据目录权限。") from None
                prepared = PreparedAudio(trace, path, record, text, settings.append_text and automatic, automatic)
                prepared.parameter_changes = log_parameter_changes(effective.provider.request, values)
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
                logger.warning(f"[TTS/发送][{prepared.trace}] 语音发送失败 | {'恢复原文' if prepared.automatic else '返回命令错误提示'}")
                # Handled after framework sending: automatic fallback or command yield.
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

    async def send_automatic_text(self, event: AstrMessageEvent, prepared: PreparedAudio) -> None:
        """Hooks cannot yield. Send the follow-up with the session's quote setting."""
        if prepared.sent:
            if not prepared.append_text:
                return
            await asyncio.sleep(ORIGINAL_DELAY)
        elif not prepared.error:
            # Another plugin discarded/replaced the audio: don't inject unexpected text.
            logger.debug(f"[TTS/发送][{prepared.trace}] 语音未被发送，跳过附带原文")
            return
        result = event.plain_result(prepared.text)
        config = self.context.get_config(umo=event.unified_msg_origin)
        if config.get("platform_settings", {}).get("reply_with_quote", False):
            result.chain.insert(0, Comp.Reply(id=event.message_obj.message_id))
        try:
            await event.send(result)
            logger.info(f"[TTS/发送][{prepared.trace}] {'原文发送成功 | 间隔≥0.5s' if prepared.sent else '失败回退原文已发送'}")
        except Exception:
            logger.warning(f"[TTS/发送][{prepared.trace}] {'附带原文' if prepared.sent else '回退原文'}发送失败，不重发语音")

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
