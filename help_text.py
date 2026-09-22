"""Plain-text session help; no generation, random sampling or secret disclosure."""
import re

from .configuration import Settings
from .request_template import references

SENSITIVE = re.compile(r"key|token|secret|password|passwd|authorization|credential|signature|密钥|密码|令牌", re.I)


def short(value: str, limit: int = 50) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[:limit] + "…"


def build_help(settings: Settings, sid: str, detailed: bool = False, prefix: str = "/",
               llm_model: str | None = None) -> str:
    effective = settings.effective(sid)
    if not effective.enabled:
        return f"TTS本会话未启用\n如需使用，请联系管理员配置会话组。\n发送 {prefix}sid 可获取本会话标识。"
    provider = effective.provider
    name = short(provider.name) if provider else "尚未选择"
    llm_probability = ("跟随全局各占位符设置" if effective.llm_probability is None
                       else f"{effective.llm_probability * 100:g}%（本组统一设置）")
    lines = [
        "TTS已启用",
        "提供商：" + name + ("（本组）" if effective.provider_override else "（全局）"),
        f"自动转换：{effective.probability * 100:g}% · 最多{effective.max_length}字 · "
        + ("仅转换LLM输出" if settings.only_llm else "转换机器人纯文本回复"),
        "语音后附原文：" + ("开启" if effective.append_text else "关闭")
        + ("（本组，仅自动转换）" if effective.append_text_override else "（全局，仅自动转换）"),
        "LLM赋值调用概率：" + llm_probability,
    ]
    if detailed:
        group = (f"第{effective.group_index + 1}组" if effective.group_index is not None
                 else "无独立分组，使用全局设置")
        if effective.remark:
            group += " · " + short(effective.remark)
        lines.extend([
            "会话组：" + group,
            f"自动转换来源：概率{'本组' if effective.probability_override else '全局'} · 长度{'本组' if effective.length_override else '全局'}",
        ])
        if provider:
            lines.append(f"生成任务并行上限：{provider.concurrency}（同提供商共享，超出排队）")
        lines.extend([
            "只转换单条纯文本回复，不处理流式或混合消息。",
            "开启附原文时，语音成功至少0.5秒后，由主框架处理原回复。",
            "LLM赋值概率用于生成情绪等参数，不是决定是否转换语音。",
        ])
    sensitive = set()
    example_names = []
    has_llm = False
    if provider:
        for key, value in provider.request.parameters:
            if SENSITIVE.search(key):
                sensitive.update(references(value))
        lines.extend(["", "当前使用的占位符"])
        for name in provider.request.used_names:
            if name == "{text}":
                detail = "朗读原文，不可覆盖"
            elif name in sensitive or SENSITIVE.search(name):
                detail = "敏感参数，值不展示"
            elif name in effective.overrides:
                value = effective.overrides[name]
                detail = (short(value) if value is not None else "无值，省略对应参数") + "（本组覆盖）"
            elif name not in settings.placeholders:
                detail = "未定义，请检查配置"
            else:
                item = settings.placeholders[name]
                kind = item["__template_key"]
                if kind == "fixed":
                    detail = (short(item.get("value")) if item.get("value") else "无值，省略对应参数") + "（全局）"
                elif kind == "random":
                    detail = f"随机{'整数' if item['mode'] == 'integer' else '小数'}，范围{item['minimum']}～{item['maximum']}"
                    if item["mode"] == "decimal":
                        detail += f"，{item['precision']}位小数"
                else:
                    has_llm = True
                    probability = item["probability"] if effective.llm_probability is None else effective.llm_probability
                    source = "全局" if effective.llm_probability is None else "本组"
                    fallback = item.get("fallback", "")
                    detail = f"LLM生成，调用概率{probability * 100:g}%（{source}）"
                    detail += "\n  未生成时：" + ("使用备用值 " + short(fallback) if fallback else "省略对应请求参数")
            lines.append(f"{name}：{detail}")
        example_names = [n for n in provider.request.used_names
                         if n != "{text}" and n not in sensitive and not SENSITIVE.search(n)][:2]
    if has_llm:
        lines.append("辅助LLM：" + ((short(llm_model) if llm_model else "模型不可用或名称读取失败")
                                  if settings.llm_id else "未配置，使用备用值或省略参数"))
    if not detailed:
        lines.extend([
            "", f"使用命令朗读：{prefix}TTS 文本",
            f"设置本组附原文：{prefix}TTS本组原文 开启/关闭/跟随",
            f"查看本组原文设置：{prefix}TTS本组原文 状态",
            f"完整用法：{prefix}TTS帮助 详细",
        ])
        return "\n".join(lines)

    lines.extend(["", "朗读与临时赋值", f"{prefix}TTS 你好"])
    if example_names:
        assignment = "&".join(f"{n}=手动输入" for n in example_names)
        lines.append(f"{prefix}TTS {assignment} 你好")
        lines.append('临时赋值仅本次有效；多个参数用 & 连接，值含空格或 & 时加双引号。')
    lines.extend([
        "", "本组原文",
        f"{prefix}TTS本组原文 开启/关闭/跟随/状态",
        "修改当前首个匹配组，仅影响实际采用该组的会话。",
        "修改仅限AstrBot管理员，QQ群管理员不等同于AstrBot管理员。",
        "", "其他命令",
        f"{prefix}TTS提供商：查看提供商",
        f"{prefix}TTS切换 名称：切换全局提供商",
        f"{prefix}TTS原文 开启/关闭/状态：管理全局附原文",
        f"{prefix}TTS清理：清理未在使用的缓存",
        f"{prefix}TTS帮助：查看简版",
        "", "说明",
        "手动TTS不受自动转换概率和长度限制，成功只发语音；LLM赋值概率仍生效。",
        "全局原文设置不影响已独立开启或关闭的组。",
        "命令权限：" + ("仅AstrBot管理员。" if settings.admin_only else "普通用户也可使用旧管理命令修改全局设置、清理缓存；本组修改始终仅限AstrBot管理员。"),
    ])
    return "\n".join(lines)
