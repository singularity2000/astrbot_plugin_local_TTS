"""Validated configuration snapshots and native Dashboard dropdown metadata."""
import copy
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from uuid import uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api import AstrBotConfig

from .request_template import ConfigError, RequestTemplate, parse_overrides, placeholder_name, references


def number(value: object, label: str, minimum: float, maximum: float, integer: bool = False) -> float | int:
    try:
        if isinstance(value, bool):
            raise ValueError()
        result = float(value)
        if not math.isfinite(result) or not minimum <= result <= maximum:
            raise ValueError()
        if integer and result != int(result):
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ConfigError(f"{label}须为 {minimum:g}～{maximum:g} 内的{'整数' if integer else '数字'}。") from None
    return int(result) if integer else result


def blank(value: object) -> bool:
    return value is None or isinstance(value, str) and not value.strip()


def prepare_schema(config: "AstrBotConfig") -> bool:
    """Update only the in-memory schema; persist generated stable IDs separately."""
    changed = False
    providers = config.get("providers", [])
    choices, labels = [""], [""]
    if isinstance(providers, list):
        for item in providers:
            if not isinstance(item, dict):
                continue
            # Previously saved GET presets use the same transport; preserve their values/IDs.
            if item.get("__template_key") in {"cosyvoice", "api7863"}:
                item["__template_key"] = "custom"
                changed = True
            if not item.get("id"):
                item["id"] = uuid4().hex
                changed = True
            identifier, name = item.get("id"), item.get("name", "")
            if isinstance(identifier, str) and isinstance(name, str) and name.strip():
                choices.append(identifier)
                labels.append(name.strip())
    schema = copy.deepcopy(config.schema)
    schema["tts_provider"]["options"] = choices
    schema["tts_provider"]["labels"] = labels
    session_select = schema["sessions"]["templates"]["session"]["items"]["provider"]
    session_select["options"] = choices.copy()
    session_select["labels"] = ["跟随全局设置", *labels[1:]]
    object.__setattr__(config, "schema", schema)
    return changed


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    request: RequestTemplate
    timeout: float
    concurrency: int = 1


@dataclass(frozen=True)
class EffectiveSettings:
    enabled: bool
    provider: Provider | None
    provider_override: bool
    remark: str
    overrides: dict[str, str | None]
    probability: float
    max_length: int
    probability_override: bool = False
    length_override: bool = False
    append_text: bool = False
    append_text_override: bool = False
    llm_probability: float | None = None
    group_index: int | None = None


@dataclass(frozen=True)
class SessionGroupSettings:
    index: int
    provider: str
    remark: str
    overrides: dict[str, str | None]
    probability: float | None
    max_length: int | None
    append_text: str
    llm_probability: float | None


class Settings:
    def __init__(self, raw: dict) -> None:
        data = copy.deepcopy(dict(raw))
        self.providers = {}
        names = set()
        for item in self._list(data, "providers"):
            identifier = item.get("id")
            name = item.get("name", "")
            if not isinstance(identifier, str) or not identifier:
                raise ConfigError("提供商缺少内部标识，请重新保存配置。")
            if not isinstance(name, str) or not name.strip() or name.strip() in names:
                raise ConfigError("提供商名称不能为空或重复。")
            if identifier in self.providers:
                raise ConfigError("提供商内部标识重复，请删除复制条目并通过添加模板重新创建。")
            if item.get("__template_key") not in {"cosyvoice", "api7863", "custom"}:
                raise ConfigError("未知提供商模板，请通过添加模板创建。")
            request = RequestTemplate.parse(item.get("url", ""))
            self.providers[identifier] = Provider(identifier, name.strip(), request,
                number(item.get("timeout", 60), "请求超时", 1, 600),
                number(item.get("concurrency", 1), "提供商任务并行上限", 1, 64, True))
            names.add(name.strip())
        self.selected = data.get("tts_provider", "")
        if not isinstance(self.selected, str) or self.selected and self.selected not in self.providers:
            raise ConfigError("全局提供商已失效，请在顶部下拉框重新选择并保存。")
        self.placeholders = {}
        for item in self._list(data, "placeholders"):
            name = placeholder_name(item.get("name", ""))
            if name in self.placeholders:
                raise ConfigError("全局占位符名称不能重复。")
            kind = item.get("__template_key")
            if kind == "fixed":
                if not isinstance(item.get("value", ""), str):
                    raise ConfigError("固定占位符的值必须为文本。")
            elif kind == "llm":
                prompt = item.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ConfigError("LLM 占位符提示词不能为空。")
                if any(ref != "{text}" for ref in references(prompt)):
                    raise ConfigError("LLM 提示词只支持 {text}，不支持占位符互相引用。")
                item["probability"] = number(item.get("probability", .5), "LLM 调用概率", 0, 1)
                if not isinstance(item.get("fallback", ""), str):
                    raise ConfigError("LLM 备用值必须为文本。")
            elif kind == "random":
                mode = item.get("mode", "integer")
                if mode not in {"integer", "decimal"}:
                    raise ConfigError("随机数类型须为整数或小数。")
                precision = 0 if mode == "integer" else number(item.get("precision", 2), "小数位数", 0, 8, True)
                try:
                    lo, hi = Decimal(str(item.get("minimum", 0))), Decimal(str(item.get("maximum", 100)))
                    if not lo.is_finite() or not hi.is_finite() or lo > hi or max(abs(lo), abs(hi)) > 10**12:
                        raise ValueError()
                    if mode == "integer" and (lo != lo.to_integral_value() or hi != hi.to_integral_value()):
                        raise ValueError()
                    scale = 10 ** precision
                    low = int((lo * scale).to_integral_value(rounding=ROUND_CEILING))
                    high = int((hi * scale).to_integral_value(rounding=ROUND_FLOOR))
                    if low > high:
                        raise ValueError()
                except (InvalidOperation, ValueError, TypeError, OverflowError):
                    raise ConfigError("随机数范围无效：下限≤上限，绝对值≤10¹²，且指定精度内至少有一个数；整数模式需整数边界。") from None
                item.update(mode=mode, minimum=str(lo), maximum=str(hi), precision=precision,
                            ticks=(low, high, precision))
            else:
                raise ConfigError("未知占位符模板。")
            self.placeholders[name] = item
        auto = data.get("auto_call_config", {})
        if not isinstance(auto, dict):
            raise ConfigError("主动调用配置格式错误。")
        self.probability = number(auto.get("send_record_probability", .8), "主动转换概率", 0, 1)
        self.max_length = number(auto.get("max_resp_text_len", 75), "自动转换长度上限", 1, 100000, True)
        self.admin_only = self._bool(data, "admin_only", False)
        self.only_llm = self._bool(data, "only_llm_response", True)
        self.append_text = self._bool(data, "append_text", False)
        self.llm_id = data.get("builtin_llm", "")
        if not isinstance(self.llm_id, str):
            raise ConfigError("插件内置 LLM 配置格式错误。")
        self.sessions = {}
        entries = self._list(data, "sessions")
        self.all_sessions = not entries
        self.duplicate_sids = 0
        for index, item in enumerate(entries):
            if item.get("__template_key") != "session":
                raise ConfigError("未知会话模板。")
            sids = item.get("sids", [])
            if not isinstance(sids, list) or any(not isinstance(s, str) for s in sids):
                raise ConfigError("会话 SID 必须为文本列表。")
            provider = item.get("provider", "")
            if not isinstance(provider, str) or provider and provider not in self.providers:
                raise ConfigError("会话引用的提供商已失效，请重新选择或改为跟随全局。")
            overrides = parse_overrides(item.get("overrides", []))
            p, length = item.get("probability", ""), item.get("max_length", "")
            append = item.get("append_text", "inherit")
            if append not in ("inherit", "on", "off"):
                raise ConfigError("本组语音后附原文须为跟随全局、开启或关闭。")
            llm_p = item.get("llm_probability", "")
            group = SessionGroupSettings(
                index=index, provider=provider, remark=str(item.get("remark", "")),
                overrides=overrides,
                probability=None if blank(p) else number(p, "会话转换概率", 0, 1),
                max_length=None if blank(length) else number(length, "会话长度上限", 1, 100000, True),
                append_text=append,
                llm_probability=None if blank(llm_p) else number(llm_p, "会话LLM赋值调用概率", 0, 1),
            )
            for sid in sids:
                sid = sid.strip()
                if not sid:
                    continue
                if sid in self.sessions:
                    self.duplicate_sids += 1
                    continue
                self.sessions[sid] = group

    @staticmethod
    def _list(data: dict, key: str) -> list[dict]:
        value = data.get(key, [])
        if not isinstance(value, list) or any(not isinstance(x, dict) for x in value):
            raise ConfigError(f"{key} 须为模板列表。")
        return value

    @staticmethod
    def _bool(data: dict, key: str, default: bool) -> bool:
        value = data.get(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{key} 须为开关。")
        return value

    def effective(self, sid: str) -> EffectiveSettings:
        group = self.sessions.get(sid)
        if group is None:
            return EffectiveSettings(
                enabled=self.all_sessions, provider=self.providers.get(self.selected),
                provider_override=False, remark="", overrides={},
                probability=self.probability, max_length=self.max_length,
                append_text=self.append_text,
            )
        return EffectiveSettings(
            enabled=True, provider=self.providers.get(group.provider or self.selected),
            provider_override=bool(group.provider), remark=group.remark,
            overrides=dict(group.overrides),
            probability=self.probability if group.probability is None else group.probability,
            max_length=self.max_length if group.max_length is None else group.max_length,
            probability_override=group.probability is not None,
            length_override=group.max_length is not None,
            append_text=self.append_text if group.append_text == "inherit" else group.append_text == "on",
            append_text_override=group.append_text != "inherit",
            llm_probability=group.llm_probability, group_index=group.index,
        )
