"""Parse before substitution: values can never introduce another URL parameter."""
import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit, urlunsplit


class ConfigError(ValueError):
    """Safe, user-facing error. Never include URLs or raw configured values."""


NAME = re.compile(r"\{([^{}\s=&]+)\}")


def placeholder_name(value: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ConfigError("占位符名称须为 {名称}，名称不能含空格、花括号、& 或 =。")
    if value == "{text}":
        raise ConfigError("{text} 是内置朗读文本，不可重复定义或覆盖。")
    return value


def references(value: str) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(m.group(0) for m in NAME.finditer(value)))
    rest = NAME.sub("", value)
    if "{" in rest or "}" in rest:
        raise ConfigError("模板中存在不完整的占位符；请使用 {名称}。")
    return names


@dataclass(frozen=True)
class RequestTemplate:
    endpoint: str
    parameters: tuple[tuple[str, str], ...]
    used_names: tuple[str, ...]

    @classmethod
    def parse(cls, template: str) -> "RequestTemplate":
        if not isinstance(template, str):
            raise ConfigError("请求 URL 必须为文本。")
        try:
            parts = urlsplit(template)
            if (parts.scheme not in {"http", "https"} or not parts.hostname
                    or parts.username is not None or parts.password is not None
                    or parts.fragment or any(c.isspace() for c in parts.netloc)):
                raise ValueError()
            _ = parts.port
            endpoint = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
            if "{" in endpoint or "}" in endpoint:
                raise ValueError()
            parameters = tuple(parse_qsl(parts.query, keep_blank_values=True,
                                         encoding="utf-8", errors="strict", max_num_fields=128))
        except (TypeError, ValueError, UnicodeError):
            raise ConfigError("请求地址须为有效 HTTP/HTTPS URL；不支持账号密码、片段或地址路径占位符。") from None
        names = []
        for key, value in parameters:
            if not key or "{" in key or "}" in key:
                raise ConfigError("请求参数名不能为空或含占位符。")
            names.extend(references(value))
        if "{text}" not in names:
            raise ConfigError("请求 URL 的查询参数值中必须包含 {text}。")
        return cls(endpoint, parameters, tuple(dict.fromkeys(names)))

    def render(self, values: dict[str, str | None]) -> list[tuple[str, str]]:
        result = []
        for key, template in self.parameters:
            names = references(template)
            if any(name not in values for name in names):
                raise ConfigError("请求 URL 引用了没有定义或赋值的占位符。")
            if any(values[name] is None for name in names):
                continue
            # Single pass: braces in user text / LLM output are never evaluated again.
            value = NAME.sub(lambda match: values[match.group(0)], template)
            result.append((key, value))
        return result


def parse_overrides(lines: list[str]) -> dict[str, str | None]:
    result = {}
    if not isinstance(lines, list):
        raise ConfigError("会话占位符覆盖必须是列表。")
    for line in lines:
        if not isinstance(line, str):
            raise ConfigError("占位符覆盖项必须是文本。")
        if not line.strip():
            continue
        name, separator, value = line.partition("=")
        name = placeholder_name(name.strip())
        if not separator or name in result:
            raise ConfigError("覆盖项应为 {名称}=值，且同一名称不能重复。")
        result[name] = value if value else None
    return result


def parse_command(body: str) -> tuple[str, dict[str, str | None]]:
    """An optional assignment block ends at the first unquoted whitespace."""
    body = body.strip()
    if body == "--" or body.startswith("-- "):
        text = body[2:].lstrip()
        if not text:
            raise ConfigError("请输入需要转换的文本。")
        return text, {}
    if not re.match(r"^\{[^{}]*\}=", body):
        if not body:
            raise ConfigError("请输入需要转换的文本。")
        return body, {}
    i, values = 0, {}
    while i < len(body):
        match = NAME.match(body, i)
        if not match or body[match.end():match.end()+1] != "=":
            raise ConfigError("参数格式错误，请使用 {名称}=值&{名称}=值 文本。")
        name = placeholder_name(match.group(0))
        if name in values:
            raise ConfigError("同一个占位符不能重复赋值。")
        i = match.end() + 1
        value, quoted = [], False
        while i < len(body):
            char = body[i]
            if char == '"':
                quoted = not quoted
            elif quoted and char == "\\" and i + 1 < len(body) and body[i+1] in {'"', "\\"}:
                i += 1
                value.append(body[i])
            elif not quoted and (char.isspace() or char == "&"):
                break
            else:
                value.append(char)
            i += 1
        if quoted:
            raise ConfigError('双引号没有闭合；值含空格或 & 时，请使用 "值"。')
        values[name] = "".join(value) or None
        if i < len(body) and body[i] == "&":
            i += 1
            if i == len(body) or body[i].isspace():
                raise ConfigError("& 后应紧接 {名称}=值，不能留空。")
            continue
        text = body[i:].lstrip()
        if not text:
            raise ConfigError("参数后请加空格，再输入需要转换的文本。")
        return text, values
    raise ConfigError("请输入需要转换的文本。")
