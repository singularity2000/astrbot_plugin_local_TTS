import asyncio
import re
import random
import aiohttp
import math
import stat
import time
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from astrbot import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Record, Plain
import astrbot.core.message.components as Comp
from astrbot.api.provider import LLMResponse

# --- 常量 ---
# 插件数据保存目录
PLUGIN_DATA_DIR = Path("./data/plugin_data/astrbot_plugin_local_TTS")
# 生成的语音文件保存目录
SAVED_AUDIO_DIR = PLUGIN_DATA_DIR / "audio"


@register(
    "astrbot_plugin_local_TTS",
    "Singularity2000",
    "通过本地部署的API（如CosyVoice3）将文本转换为语音，支持多种音色和情绪。",
    "1.0.0",
    "https://github.com/Singularity2000/astrbot_plugin_local_TTS",
)
class LocalTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.auto_clean_task: Optional[asyncio.Task] = None
        self._cache_in_use = {}
        self._terminated = False
        
        # 从全局配置加载管理员列表
        bot_config = self.context.get_config()
        self.admins: List[str] = [str(admin) for admin in bot_config.get("admins_id", [])]
        self.admin_only: bool = False

        self.load_config()

    def load_config(self):
        """从配置对象中加载和解析配置"""
        base_setting: Dict = self.config.get("base_setting", {})
        self.api_url: str = base_setting.get("api_url", "http://localhost:9880")

        auto_call_config: Dict = self.config.get("auto_call_config", {})
        self.send_record_probability: float = auto_call_config.get("send_record_probability", 0.4)
        self.max_resp_text_len: int = auto_call_config.get("max_resp_text_len", 75)

        emotion_config: Dict = self.config.get("emotion_config", {})
        self.emotion_llm_id: str = emotion_config.get("emotion_llm", "")
        self.emotion_probability: float = emotion_config.get("emotion_probability", 0.5)

        self.enabled_sessions: list = self.config.get("enabled_sessions", [])
        self.global_speaker: str = self.config.get("global_speaker", "")
        self.admin_only: bool = self.config.get("admin_only", False)
        self.only_llm_response: bool = self.config.get("only_llm_response", False)
        self.cache_limits = {}
        self.cache_config_valid = True
        cache_config = self.config.get("cache_management", {})
        if not isinstance(cache_config, dict):
            cache_config = {}
            self.cache_config_valid = False
            logger.error("缓存管理配置格式错误，自动清理已暂停。")
        for key, scale in (("max_age_hours", 3600), ("max_size_mb", 1024 * 1024), ("max_files", 1)):
            try:
                self.cache_limits[key] = self._parse_cache_limit(cache_config.get(key), key, scale)
            except ValueError:
                self.cache_config_valid = False
                logger.error(f"缓存配置 {key} 无效：应留空或填写正数（数量须为正整数），自动清理已暂停。")
        self.cache_auto_enabled = self.cache_config_valid and any(
            value is not None for value in self.cache_limits.values()
        )

        logger.info("本地TTS插件配置已加载。")
        if self.auto_clean_task:
            self.auto_clean_task.cancel()
            self.auto_clean_task = None
        if self.cache_auto_enabled and not self._terminated:
            self.auto_clean_task = asyncio.create_task(self._auto_clean_task())

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
        directory = SAVED_AUDIO_DIR.absolute()
        try:
            for path in (directory, *directory.parents):
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    continue
                if self._is_link(info) or not stat.S_ISDIR(info.st_mode):
                    logger.error("缓存目录或其父目录不是普通目录，已跳过缓存操作。")
                    return False
        except OSError:
            logger.warning("无法检查缓存目录，已跳过缓存操作。")
            return False
        return True

    def _clean_cache(self, force: bool = False) -> Tuple[int, float, int]:
        """按约束淘汰；force 手动清空。返回成功删除数、MB、在用跳过数。

        扫描、写入和保护登记均在事件循环中同步执行且不 await，避免清理相互穿插。
        """
        if not force and not self.cache_auto_enabled:
            return 0, 0.0, 0
        if not self._cache_directory_safe():
            return 0, 0.0, 0
        files = []
        try:
            for path in SAVED_AUDIO_DIR.iterdir():
                try:
                    info = path.lstat()
                    if stat.S_ISREG(info.st_mode) and not self._is_link(info):
                        files.append((path, info))
                except FileNotFoundError:
                    continue
                except OSError:
                    # 不能完整统计时，不据此做破坏性清理。
                    logger.warning("无法读取缓存文件信息，本轮清理已跳过。")
                    return 0, 0.0, 0
        except FileNotFoundError:
            return 0, 0.0, 0
        except OSError:
            logger.warning("无法扫描缓存目录，本轮清理已跳过。")
            return 0, 0.0, 0

        files.sort(key=lambda entry: (entry[1].st_mtime, entry[0].name))
        remaining_count = len(files)
        remaining_bytes = sum(info.st_size for _, info in files)
        max_age = self.cache_limits.get("max_age_hours")
        max_size = self.cache_limits.get("max_size_mb")
        max_count = self.cache_limits.get("max_files")
        now = time.time()
        count = removed_bytes = skipped = 0

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
                logger.warning(f"无法删除缓存文件 {path.name}，将在后续清理时重试。")
                continue
            count += 1
            removed_bytes += info.st_size
            remaining_count -= 1
            remaining_bytes -= info.st_size

        if not force and (over_capacity() or expired_remaining):
            logger.warning("缓存仍有超限或过期文件（可能正在使用或无法删除），将在后续清理时重试。")
        return count, removed_bytes / (1024 * 1024), skipped

    def _run_cache_cleanup(self) -> None:
        if self._terminated:
            return
        count, size_mb, _ = self._clean_cache()
        if count:
            logger.info(f"TTS缓存已清理{count}个文件，共{size_mb:.2f}MB。")

    def _protect_audio(self, event: AstrMessageEvent, path: Path) -> None:
        task = asyncio.current_task()
        # 发送完成钩子未执行（失败、取消、结果被其他插件丢弃）时，
        # 等所属流水线任务结束后释放，不用超时猜测文件是否仍在使用。
        def task_done(_):
            self._release_audio(path)
            self._run_cache_cleanup()
        self._cache_in_use[path] = (event, task, task_done)
        if task is not None:
            task.add_done_callback(task_done)

    def _release_audio(self, path: Path) -> None:
        entry = self._cache_in_use.pop(path, None)
        if entry is not None:
            _, task, callback = entry
            if task is not None:
                task.remove_done_callback(callback)

    @filter.after_message_sent()
    async def after_tts_sent(self, event: AstrMessageEvent) -> None:
        paths = [path for path, entry in self._cache_in_use.items() if entry[0] is event]
        for path in paths:
            self._release_audio(path)
        if paths:
            self._run_cache_cleanup()

    def _get_session_info(self, event: AstrMessageEvent) -> Tuple[Optional[str], Optional[str]]:
        """
        获取当前会话的(音色, 指令)配置。
        通过遍历配置列表并查找精确匹配的SID来获取配置。
        """
        event_sid = event.unified_msg_origin
        
        # 如果白名单为空，则所有会话都启用，并使用全局音色
        if not self.enabled_sessions:
            return (self.global_speaker or None, None)

        # 遍历配置列表，查找匹配的会话
        for session_config_str in self.enabled_sessions:
            session_config_str = session_config_str.strip()
            if not session_config_str:
                continue

            # 从配置项中提取SID部分进行精确匹配
            # 'napcat:GroupMessage:123456:speaker:instruct' -> 'napcat:GroupMessage:123456'
            parts = session_config_str.split(':')
            if len(parts) >= 3:
                config_sid = ':'.join(parts[:3])
                if config_sid == event_sid:
                    # SID完全匹配，现在解析音色和指令
                    full_parts = session_config_str.split(':', 4)
                    speaker = full_parts[3] if len(full_parts) > 3 and full_parts[3] else self.global_speaker
                    instruct = full_parts[4] if len(full_parts) > 4 and full_parts[4] else None
                    
                    logger.debug(f"会话 {event_sid} 匹配成功: '{session_config_str}', "
                                 f"音色: {speaker}, 指令: {instruct}")
                    return (speaker or None, instruct)
            elif session_config_str == event_sid:
                # 处理只有SID，没有其他参数的情况
                logger.debug(f"会话 {event_sid} 匹配成功: '{session_config_str}', "
                             f"使用全局音色。")
                return (self.global_speaker or None, None)


        # 如果遍历完都未找到匹配项
        logger.debug(f"会话 {event_sid} 未在启用的会话列表中找到匹配项。")
        return ("NOT_FOUND", None)


    async def _make_tts_request(self, text: str, speaker: str, instruct: Optional[str]) -> Optional[bytes]:
        """向TTS API发送请求并以字节形式返回音频内容"""
        if not self.api_url:
            logger.error("TTS API的URL未配置！")
            return None
        
        params = {"text": text, "speaker": speaker}
        if instruct:
            params["instruct"] = instruct

        try:
            async with aiohttp.ClientSession() as session:
                logger.debug(f"发送TTS请求: URL={self.api_url}, 参数={params}")
                async with session.get(self.api_url, params=params, timeout=60) as response:
                    if response.status == 200:
                        content = await response.read()
                        if content and not content.strip().startswith(b"{"):
                           return content
                        else:
                           logger.warning(f"TTS API返回了错误信息: {content.decode('utf-8', 'ignore')}")
                           return None
                    else:
                        error_content = await response.text()
                        logger.warning(
                            f"TTS API 请求失败, 状态码: {response.status}, "
                            f"参数: {params}, 错误信息: {error_content}"
                        )
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"TTS API 请求超时. 参数: {params}")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"TTS API 请求时发生网络错误: {e}. 参数: {params}")
            return None

    async def _generate_speech(
        self, 
        event: AstrMessageEvent, 
        text: str, 
        override_speaker: Optional[str] = None, 
        override_instruct: Optional[str] = None
    ) -> Optional[str]:
        """
        生成语音的核心逻辑。
        处理会话配置、情绪检测和API调用。
        返回保存的音频文件的路径，如果失败则返回错误代码或None。
        """
        session_speaker, session_instruct = self._get_session_info(event)

        # 核心授权检查：如果会话不在白名单中，则静默退出。
        if session_speaker == "NOT_FOUND":
            logger.debug(f"会话 {event.unified_msg_origin} 不在启用的会话白名单中，TTS指令或自动调用被忽略。")
            return None

        # 确定最终使用的音色。优先级：指令强制 > 会话配置 > 全局配置
        final_speaker = override_speaker or session_speaker or self.global_speaker
        if not final_speaker:
            # 此情况发生于：会话在白名单内但未指定音色，且全局音色也为空。
            logger.warning(f"当前会话 {event.unified_msg_origin} 未找到可用音色（指令、会话、全局均未配置）。")
            return "no_speaker"

        # 组合指令: 指令强制 > 会话配置
        final_instruct = override_instruct if override_instruct is not None else session_instruct
        
        # --- LLM情绪判断逻辑 ---
        if override_instruct is None and self.emotion_llm_id and self.emotion_probability > 0 and random.random() < self.emotion_probability:
            llm_provider = self.context.get_provider_by_id(self.emotion_llm_id)
            if not llm_provider:
                logger.warning(f"未找到配置的情绪判断LLM: {self.emotion_llm_id}")
            else:
                prompt = f"判断“{text}”这句话的情绪，只返回一个词，不要加任何多余说明。"
                try:
                    logger.info(f"调用LLM判断情绪, prompt: {prompt}")
                    resp: LLMResponse = await llm_provider.text_chat(prompt=prompt)
                    emotion_word = resp.completion_text.strip()
                    if emotion_word:
                        logger.info(f"LLM判断情绪为: {emotion_word}")
                        final_instruct = f"{final_instruct.strip()}, {emotion_word}" if final_instruct and final_instruct.strip() else emotion_word
                except Exception as e:
                    logger.warning(f"调用LLM判断情绪失败: {e}")

        # --- API调用 ---
        audio_bytes = await self._make_tts_request(text=text, speaker=final_speaker, instruct=final_instruct)
        if not audio_bytes:
            logger.warning(f"音色配置或模型可能存在问题，跳过生成: speaker={final_speaker}, instruct={final_instruct}")
            return "api_fail"

        # --- 保存音频文件 ---
        sanitized_text = re.sub(r'[\\/*?:"<>|\x00-\x1f]', "", text)[:20]
        # 每次生成独立文件，避免并发请求覆盖尚未发送的音频。
        file_name = f"tts_{uuid.uuid4().hex}_{sanitized_text}.wav"
        save_path = SAVED_AUDIO_DIR / file_name
        
        if not self._cache_directory_safe():
            return None
        try:
            SAVED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            self._protect_audio(event, save_path)
            with open(save_path, "xb") as f:
                f.write(audio_bytes)
            logger.info(f"成功生成语音文件: {save_path}")
            return str(save_path)
        except IOError as e:
            self._release_audio(save_path)
            logger.error(f"保存语音文件失败: {e}")
            return None

    @filter.on_llm_response()
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """标记当前事件为 LLM 回复"""
        setattr(event, "_is_llm_out", True)

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """根据概率将发出的文本转为语音"""
        if self.only_llm_response and not getattr(event, "_is_llm_out", False):
            return

        if self.send_record_probability <= 0 or random.random() > self.send_record_probability:
            return

        chain = event.get_result().chain
        if not (len(chain) == 1 and isinstance(chain[0], Comp.Plain)):
            return

        text_to_send = chain[0].text
        if len(text_to_send) > self.max_resp_text_len:
            return

        save_path_or_error = await self._generate_speech(event, text_to_send)

        if save_path_or_error and save_path_or_error not in ["no_speaker", "api_fail"]:
            chain.clear()
            chain.append(Record.fromFileSystem(save_path_or_error))

    @filter.command("TTS")
    async def on_tts_command(self, event: AstrMessageEvent):
        """
        处理TTS指令, 格式: TTS 【instruct】【speaker】...text...
        """
        # 管理员权限检查
        if self.admin_only and str(event.get_sender_id()) not in self.admins:
            session_speaker, _ = self._get_session_info(event)
            # 如果会话在白名单内，则提示无权限
            if session_speaker != "NOT_FOUND":
                # 直接发送消息，而不是yield，以绕过on_decorating_result钩子
                await event.send(event.plain_result("此命令仅限管理员使用。"))
            # 如果不在白名单，则静默忽略
            return
            
        msg_text = event.get_message_str().strip()

        # 移除指令前缀"TTS"，为正则解析做准备
        if msg_text.upper().startswith("TTS"):
            msg_text = msg_text[3:].strip()
        
        override_speaker = None
        override_instruct = None
        text_part = msg_text

        # 严格匹配句首的【instruct】【speaker】...text... 格式
        two_tags_match = re.match(r"^\s*【([^】]*)】\s*【([^】]*)】(.*)", msg_text, re.DOTALL)
        if two_tags_match:
            override_instruct = two_tags_match.group(1).strip()
            override_speaker = two_tags_match.group(2).strip()
            text_part = two_tags_match.group(3).strip()
        else:
            # 严格匹配句首的【instruct】...text... 格式
            one_tag_match = re.match(r"^\s*【([^】]*)】(.*)", msg_text, re.DOTALL)
            if one_tag_match:
                override_instruct = one_tag_match.group(1).strip()
                text_part = one_tag_match.group(2).strip()

        if not text_part:
            yield event.plain_result("请输入需要转换的文本。")
            return
        
        save_path_or_error = await self._generate_speech(
            event, text_part, override_speaker=override_speaker, override_instruct=override_instruct
        )

        if save_path_or_error == "no_speaker":
            yield event.plain_result("TTS失败，当前会话未配置音色或配置有误，详情查看日志。")
        elif save_path_or_error == "api_fail":
            yield event.plain_result("TTS失败，音色配置或API有误，详情查看日志。")
        elif save_path_or_error:
            yield event.chain_result([Record.fromFileSystem(save_path_or_error)])
        else:
            # 如果 save_path_or_error 为 None，则说明会话不在白名单。
            logger.info(f"收到来自会话 {event.unified_msg_origin} 的TTS指令，但该会话不在白名单内，已静默处理。")

    @filter.command("TTS清理")
    async def on_clean_cache_command(self, event: AstrMessageEvent):
        """手动清理TTS缓存文件。"""
        count, size_mb, skipped = self._clean_cache(force=True)
        response_text = f"TTS缓存已清理{count}个文件，共{size_mb:.2f}MB。"
        if skipped:
            response_text += f"跳过{skipped}个正在使用的文件。"
        yield event.plain_result(response_text)

    async def _auto_clean_task(self):
        """启动时检查一次，之后每分钟检查；三项全空时不创建本任务。"""
        try:
            while True:
                try:
                    self._run_cache_cleanup()
                except Exception:
                    logger.exception("TTS缓存自动清理任务出现错误。")
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass

    async def terminate(self):
        self._terminated = True
        logger.info("本地TTS插件已终止。")
        if self.auto_clean_task:
            self.auto_clean_task.cancel()
            try:
                await self.auto_clean_task
            except asyncio.CancelledError:
                pass # 预期中的异常

        for path in list(self._cache_in_use):
            self._release_audio(path)
        self.auto_clean_task = None
