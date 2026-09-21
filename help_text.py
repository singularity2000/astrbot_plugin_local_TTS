"""Compact, side-effect-free help, with credential-bearing placeholders redacted."""
import re

from .configuration import Settings
from .request_template import references

SENSITIVE = re.compile(r"key|token|secret|password|authorization|credential|密钥|密码|令牌", re.I)


def short(value: str, limit: int = 50) -> str:
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[:limit] + "…"


def build_help(settings: Settings, sid: str) -> str:
    effective = settings.effective(sid)
    lines = ["【TTS帮助 · 2.0.0】", "本会话：" + ("已启用" if effective.enabled else "未启用，TTS不会生成语音")]
    if effective.remark:
        lines.append("备注：" + short(effective.remark))
    if effective.enabled:
        name = effective.provider.name if effective.provider else "尚未选择"
        lines.append("提供商：" + name + ("（会话指定）" if effective.provider_override else "（跟随全局）"))
        lines.append(f"自动转换：{effective.probability * 100:g}%（{'会话覆盖' if effective.probability_override else '全局'}）｜最多{effective.max_length}字（{'会话覆盖' if effective.length_override else '全局'}）")
    lines.extend([
        "转换对象：" + ("仅LLM回复" if settings.only_llm else "机器人纯文本回复") + "；仅处理单个纯文本消息",
        "自动转换后附原文：" + ("开启" if settings.append_text else "关闭"),
        "命令权限：" + ("仅管理员" if settings.admin_only else "所有用户（可修改全局设置）"),
    ])
    if effective.enabled and effective.provider:
        lines.append("请求使用的占位符：")
        sensitive = set()
        for key, value in effective.provider.request.parameters:
            if SENSITIVE.search(key):
                sensitive.update(references(value))
        for name in effective.provider.request.used_names:
            if name == "{text}":
                detail = "本次朗读文本，不可修改"
            elif name in sensitive or SENSITIVE.search(name):
                detail = "敏感参数，值不展示"
            elif name in effective.overrides:
                value = effective.overrides[name]
                detail = (short(value) if value is not None else "无值，省略对应参数") + "（会话覆盖）"
            elif name not in settings.placeholders:
                detail = "未定义，请检查配置"
            else:
                item = settings.placeholders[name]
                kind = item["__template_key"]
                if kind == "fixed":
                    detail = (short(item.get("value")) if item.get("value") else "无值，省略对应参数") + "（全局）"
                elif kind == "random":
                    detail = f"随机{'整数' if item['mode'] == 'integer' else '小数'} {item['minimum']}～{item['maximum']}"
                    if item["mode"] == "decimal":
                        detail += f"，{item['precision']}位小数"
                else:
                    fallback = item.get("fallback", "")
                    detail = f"LLM生成，概率{item['probability'] * 100:g}%；未生成时" + ("用备用值：" + short(fallback) if fallback else "不传对应参数")
            lines.append(f"• {name}：{detail}")
        lines.append("内置LLM：" + (short(settings.llm_id) if settings.llm_id else "未配置（LLM占位符用备用值或省略）"))
    example_names = []
    if effective.enabled and effective.provider:
        example_names = [name for name in effective.provider.request.used_names
                         if name != "{text}" and name not in sensitive and not SENSITIVE.search(name)][:2]
    assignment = "&".join(f"{name}=手动输入" for name in example_names)
    lines.extend([
        "用法：TTS 你好",
        (f"临时赋值：TTS {assignment} 你好" if assignment else "临时赋值格式：TTS {名称}=值 文本"),
        '请使用上方实际占位符名称；多个参数用 & 连接，值含空格或 & 时加双引号。只影响这一次。',
        "TTS提供商｜TTS切换 名称｜TTS原文 开启/关闭/状态｜TTS清理",
        "管理命令回复保持文字；手动TTS不受自动概率和长度限制。",
    ])
    return "\n".join(lines)
