"""Shared HTTP transport, bounded audio responses and format detection."""
import json
import re
import time
from urllib.parse import unquote_plus

import aiohttp
from astrbot.api import logger

from .configuration import Provider
from .request_template import RequestTemplate, references


_SENSITIVE = re.compile(r"key|token|secret|password|passwd|authorization|credential|signature|密钥|密码|令牌", re.I)


def log_url(url: str) -> str:
    """Preserve the wire URL verbatim except credential-bearing query values."""
    base, separator, query = url.partition("?")
    if not separator:
        return base
    safe = []
    for part in query.split("&"):
        key, equals, value = part.partition("=")
        safe.append(key + equals + ("REDACTED" if _SENSITIVE.search(unquote_plus(key)) else value))
    return base + "?" + "&".join(safe)


def log_parameter_changes(template: RequestTemplate, values: dict[str, str | None]) -> str:
    """Describe resolved substitutions only; retain neither credentials nor raw controls."""
    sensitive = {
        name
        for key, value in template.parameters if _SENSITIVE.search(key)
        for name in references(value)
    }
    fields = []
    for name in template.used_names:
        value = values[name]
        if name == "{text}" or value == name:
            continue
        if name in sensitive or _SENSITIVE.search(name):
            display = "[已脱敏]"
        elif value is None:
            display = "[已省略]"
        else:
            display = value
        # Strip JSON's enclosing quotes, but preserve escaping for single-line logs.
        label = json.dumps(name, ensure_ascii=False)[1:-1]
        display = json.dumps(display, ensure_ascii=False)[1:-1]
        fields.append(f"{label}={display}")
    return "".join(f" | {field}" for field in fields)

class TTSError(Exception):
    """Safe user-facing synthesis/transport failure."""


def audio_suffix(data: bytes) -> str:
    if len(data) >= 12 and data[:4] in {b"RIFF", b"RF64"} and data[8:12] == b"WAVE":
        return ".wav"
    if data.startswith(b"OggS"):
        return ".ogg"
    if data.startswith(b"fLaC"):
        return ".flac"
    if data.startswith(b"ID3"):
        return ".mp3"
    if len(data) >= 2 and data[0] == 0xff:
        if data[1] & 0xf6 == 0xf0:
            return ".aac"
        if data[1] & 0xe0 == 0xe0 and data[1] & 0x06:
            return ".mp3"
    raise TTSError("接口未返回可识别的音频（支持 WAV/MP3/OGG/FLAC/AAC）；请确认不是 JSON 地址或错误页面。")


class TTSClient:
    MAX_BYTES = 32 * 1024 * 1024

    def __init__(self) -> None:
        self._session = None

    async def synthesize(self, provider: Provider, parameters: list[tuple[str, str]], trace: str = "-") -> tuple[bytes, str]:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        started = time.monotonic()
        try:
            async with self._session.get(provider.request.endpoint, params=parameters,
                    timeout=aiohttp.ClientTimeout(total=provider.timeout),
                    allow_redirects=False) as response:
                if response.status != 200:
                    raise TTSError(f"TTS 接口返回 HTTP {response.status}，请检查服务和参数；不自动跟随重定向。")
                if response.content_length and response.content_length > self.MAX_BYTES:
                    raise TTSError("音频超过 32 MB 安全上限。")
                content = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    content.extend(chunk)
                    if len(content) > self.MAX_BYTES:
                        raise TTSError("音频超过 32 MB 安全上限。")
                data = bytes(content)
                suffix = audio_suffix(data)
                # request_info carries aiohttp's actual encoded URL, including duplicate keys.
                url = log_url(str(response.request_info.real_url))
                logger.debug(f"[TTS/GET][{trace}] 200 | {time.monotonic() - started:.2f}s | {len(data) / (1024 * 1024):.3f}MB | {url}")
                return data, suffix
        except TTSError:
            raise
        except TimeoutError:
            raise TTSError("TTS 请求超时，请检查服务或增加提供商请求超时。") from None
        except (aiohttp.ClientError, ValueError, OSError):
            raise TTSError("无法请求 TTS 服务，请检查地址、端口及服务状态。") from None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
