"""AstrBot Local TTS 2.0: hooks and commands only; domain logic lives in modules."""
import asyncio
import copy
import random
from uuid import uuid4
from collections.abc import AsyncIterator

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageEventResult, ResultContentType, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star, StarTools, register
import astrbot.api.message_components as Comp

from .audio_cache import AudioCache, format_size
from .configuration import Settings, prepare_schema
from .delivery import Delivery
from .request_template import ConfigError, parse_command
from .tts_client import TTSError


@register("astrbot_plugin_local_TTS", "Singularity2000",
          "通过 GET 请求模板与占位符连接本地 TTS，支持会话覆盖和内置 LLM。",
          "2.1.0", "https://github.com/Singularity2000/astrbot_plugin_local_TTS")
class LocalTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.settings = None
        self.config_error = "插件尚未初始化。"
        self.cache = None
        self.delivery = None
        self._config_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Use the framework lifecycle; do not start tasks inside __init__."""
        if prepare_schema(self.config):
            try:
                self.config.save_config()
            except Exception:
                self.config_error = "提供商内部标识保存失败，请检查配置目录权限后重载。"
                logger.warning("[TTS/配置] " + self.config_error)
                return
        self._load_settings()
        try:
            directory = StarTools.get_data_dir("astrbot_plugin_local_TTS") / "audio"
            self.cache = AudioCache(directory, self.config.get("cache_management", {}))
            self.cache.start()
            self.delivery = Delivery(self.context, self.cache)
        except Exception:
            self.config_error = "无法初始化音频缓存，请检查插件数据目录权限。"
            self.settings = None
            logger.warning("[TTS/配置] " + self.config_error)

    def _load_settings(self) -> None:
        try:
            self.settings = Settings(self.config)
            self.config_error = ""
            if self.settings.duplicate_sids:
                logger.warning(f"[TTS/配置] 发现{self.settings.duplicate_sids}处重复SID，按从上到下首个匹配组生效；调整组顺序会改变设置。")
            selected = self.settings.providers.get(self.settings.selected)
            logger.info(f"[TTS/配置] 已加载 | 提供商={len(self.settings.providers)}个 | 全局={'已选择' if selected else '未选择'} | 自动概率={self.settings.probability:g} | 仅LLM={self.settings.only_llm}")
        except ConfigError as exc:
            self.settings = None
            self.config_error = str(exc)
            logger.warning(f"[TTS/配置] 无效，生成暂停：{exc}")

    def _reply(self, event: AstrMessageEvent, text: str) -> MessageEventResult:
        event.set_extra("local_tts_control", True)
        return event.plain_result(text)

    def _command_gate(self, event: AstrMessageEvent, ready: bool = True) -> tuple[bool, str]:
        event.set_extra("local_tts_control", True)
        admin_only = self.settings.admin_only if self.settings else self.config.get("admin_only", True) is not False
        if admin_only and not event.is_admin():
            enabled = self.settings and self.settings.effective(event.unified_msg_origin).enabled
            return False, "此命令仅限管理员使用。" if enabled else ""
        if ready and (self.settings is None or self.delivery is None):
            return False, "TTS 暂不可用：" + self.config_error
        return True, ""

    @staticmethod
    def _body(event: AstrMessageEvent, command: str) -> str:
        # AstrBot removes the wake prefix before command matching. Preserve original whitespace.
        message = event.get_message_str().strip()
        return message[len(command):].lstrip()

    def _commit_settings(self, candidate: dict) -> None:
        """Caller holds _config_lock; publish only a successfully saved snapshot."""
        settings = Settings(candidate)
        previous = copy.deepcopy(dict(self.config))
        try:
            self.config.save_config(candidate)
        except Exception:
            self.config.clear()
            self.config.update(previous)
            raise ConfigError("保存配置失败，当前运行设置未改变。") from None
        self.settings = settings

    async def _save_option(self, key: str, value: str | bool) -> None:
        async with self._config_lock:
            candidate = copy.deepcopy(dict(self.config))
            candidate[key] = value
            self._commit_settings(candidate)
            action = "全局提供商已切换" if key == "tts_provider" else f"语音后附原文={'开启' if value else '关闭'}"
            logger.info(f"[TTS/命令] {action} | 已保存")

    async def _save_group_original(self, sid: str, value: str) -> None:
        async with self._config_lock:
            candidate = copy.deepcopy(dict(self.config))
            group = Settings(candidate).sessions.get(sid)
            if group is None:
                raise ConfigError("本会话没有所属会话组，请先在插件设置中配置；未修改全局设置。")
            candidate["sessions"][group.index]["append_text"] = value
            self._commit_settings(candidate)
            logger.info(f"[TTS/命令] 第{group.index + 1}组附原文设置已保存")

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse) -> None:
        event.set_extra("local_tts_llm_response", True)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent) -> None:
        settings = self.settings
        delivery = self.delivery
        if settings is None or delivery is None or event.get_extra("local_tts_control", False):
            return
        if settings.only_llm and not event.get_extra("local_tts_llm_response", False):
            logger.debug("[TTS/跳过] 非LLM回复")
            return
        result = event.get_result()
        if result is None or len(result.chain) != 1 or not isinstance(result.chain[0], Comp.Plain):
            return
        if result.result_content_type in {ResultContentType.STREAMING_RESULT, ResultContentType.STREAMING_FINISH}:
            return
        text = result.chain[0].text
        effective = settings.effective(event.unified_msg_origin)
        reason = ("会话未启用" if not effective.enabled else "空文本" if not text.strip()
                  else "文本超长" if len(text) > effective.max_length
                  else "未命中概率" if random.random() >= effective.probability else "")
        if reason:
            logger.debug(f"[TTS/跳过] {reason}")
            return
        trace = uuid4().hex[:8]
        try:
            prepared = await delivery.prepare(event, settings, effective, text, {}, True, trace)
            await delivery.send_automatic(event, prepared, result)
        except (ConfigError, TTSError) as exc:
            logger.warning(f"[TTS/失败][{trace}] 自动转换：{exc}；原回复交由主框架处理")
        except Exception:
            logger.warning(f"[TTS/失败][{trace}] 自动转换：内部错误，原回复交由主框架处理")

    @filter.command("TTS")
    async def on_tts_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """TTS [{名称}=值&{名称}=值] 要朗读的文本；复杂值可用双引号。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        settings = self.settings
        effective = settings.effective(event.unified_msg_origin)
        if not effective.enabled:
            return
        trace = uuid4().hex[:8]
        prepared = None
        error = ""
        try:
            text, overrides = parse_command(self._body(event, "TTS"))
            prepared = await self.delivery.prepare(event, settings, effective, text, overrides, False, trace)
        except (ConfigError, TTSError) as exc:
            error = str(exc)
        except Exception:
            error = "内部错误，请检查插件配置及兼容性。"
        if error:
            logger.warning(f"[TTS/失败][{trace}] 手动生成：{error}")
            yield self._reply(event, "TTS失败：" + error)
            return
        try:
            yield event.chain_result([prepared.record])
            if prepared.error:
                yield self._reply(event, "TTS失败：" + prepared.error)
        finally:
            self.delivery.finish(event, prepared)

    @filter.command("TTS提供商")
    async def providers_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """列出已保存的提供商、全局默认及当前会话实际使用项。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        effective = self.settings.effective(event.unified_msg_origin)
        lines = ["【TTS提供商】"]
        for identifier, provider in self.settings.providers.items():
            flags = []
            if identifier == self.settings.selected:
                flags.append("全局默认")
            if effective.enabled and effective.provider == provider:
                flags.append("当前会话")
            lines.append(provider.name + ("（" + "、".join(flags) + "）" if flags else ""))
        if not self.settings.providers:
            lines.append("尚未添加，请在插件设置中添加提供商并保存。")
        elif not self.settings.selected:
            lines.append("全局默认尚未选择。")
        if not effective.enabled:
            lines.append("本会话未启用 TTS。")
        lines.append("切换全局默认：TTS切换 提供商名称")
        yield self._reply(event, "\n".join(lines))

    @filter.command("TTS切换")
    async def switch_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """按完整名称切换并保存全局默认提供商，不影响独立指定的会话。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        name = self._body(event, "TTS切换").strip()
        selected = next((p for p in self.settings.providers.values() if p.name == name), None)
        if selected is None:
            yield self._reply(event, "未找到该提供商。请用 TTS提供商 查看名称；格式：TTS切换 名称")
            return
        try:
            await self._save_option("tts_provider", selected.id)
        except ConfigError as exc:
            yield self._reply(event, str(exc))
            return
        yield self._reply(event, f"全局提供商已切换为：{selected.name}。独立指定提供商的会话不受影响。")

    @filter.command("TTS原文")
    async def original_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """开启、关闭或查看全局语音后附原文设置。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        action = self._body(event, "TTS原文").strip()
        if action in {"开启", "关闭"}:
            try:
                await self._save_option("append_text", action == "开启")
            except ConfigError as exc:
                yield self._reply(event, str(exc))
                return
        elif action not in {"", "状态"}:
            yield self._reply(event, "用法：TTS原文 开启 / 关闭 / 状态")
            return
        yield self._reply(event, "语音后附原文：" + ("开启" if self.settings.append_text else "关闭") + "（全局，仅自动转换）")

    @filter.command("TTS本组原文")
    async def group_original_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """查看本组原文设置；AstrBot管理员可开启、关闭或恢复跟随全局。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        action = self._body(event, "TTS本组原文").strip()
        actions = {"开启": "on", "关闭": "off", "跟随": "inherit"}
        if action not in {"", "状态", *actions}:
            yield self._reply(event, "用法：TTS本组原文 开启 / 关闭 / 跟随 / 状态")
            return
        if action in actions and not event.is_admin():
            yield self._reply(event, "修改整个会话组仅限 AstrBot 管理员，QQ群管理员不等同于 AstrBot 管理员。")
            return
        sid = event.unified_msg_origin
        if sid not in self.settings.sessions:
            yield self._reply(event, "本会话没有所属会话组，请先在插件设置中配置；未修改全局设置。")
            return
        if action in actions:
            try:
                await self._save_group_original(sid, actions[action])
            except ConfigError as exc:
                yield self._reply(event, str(exc))
                return
        effective = self.settings.effective(sid)
        count = sum(g.index == effective.group_index for g in self.settings.sessions.values())
        from .help_text import short
        group_name = (short(effective.remark) + "（" if effective.remark else "") + f"第{effective.group_index + 1}组" + ("）" if effective.remark else "")
        state = "开启" if effective.append_text else "关闭"
        source = "本组指定" if effective.append_text_override else "跟随全局"
        yield self._reply(event, f"会话组：{group_name}\n语音后附原文：{state}（{source}，仅自动转换）\n影响本组 {count} 个生效会话。" + ("已保存。" if action in actions else ""))

    @filter.command("TTS清理")
    async def clean_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """清理音频缓存，跳过正在使用的文件。"""
        allowed, message = self._command_gate(event, ready=False)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        if self.cache is None:
            yield self._reply(event, "缓存尚未初始化。")
            return
        count, size, skipped = self.cache.clean()
        yield self._reply(event, f"已清理 {count} 个文件，共 {format_size(size)}；跳过 {skipped} 个在用文件。")

    @filter.command("TTS帮助")
    async def help_command(self, event: AstrMessageEvent) -> AsyncIterator[MessageEventResult]:
        """查看本会话实际设置和紧凑用法；不会调用 LLM 或生成随机数。"""
        allowed, message = self._command_gate(event)
        if not allowed:
            if message:
                yield self._reply(event, message)
            return
        from .help_text import build_help
        action = self._body(event, "TTS帮助").strip()
        if action not in {"", "详细"}:
            yield self._reply(event, "用法：TTS帮助 或 TTS帮助 详细")
            return
        config = self.context.get_config(umo=event.unified_msg_origin)
        prefixes = config.get("wake_prefix", [])
        prefix = next((p for p in prefixes if isinstance(p, str) and p), "") if isinstance(prefixes, list) else ""
        llm_model = None
        if self.settings.llm_id and self.settings.effective(event.unified_msg_origin).enabled:
            try:
                provider = self.context.get_provider_by_id(self.settings.llm_id)
                model = provider.get_model() if provider is not None else None
                if isinstance(model, str) and model.strip():
                    llm_model = model.strip()
            except Exception:
                # Display a safe fallback; never expose provider errors or configuration.
                pass
        yield self._reply(event, build_help(self.settings, event.unified_msg_origin,
                                          detailed=action == "详细", prefix=prefix,
                                          llm_model=llm_model))

    async def terminate(self) -> None:
        if self.delivery:
            await self.delivery.close()
        if self.cache:
            await self.cache.close()
