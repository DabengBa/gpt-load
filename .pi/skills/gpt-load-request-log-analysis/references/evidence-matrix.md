# 证据矩阵与分析规则

## 每个 request-log attempt 一行，关联 debug-capture attempt

| 证据 | 必须存在 |
|---|---|
| logical attempt ID / sequence | 是 |
| Provider/channel | 是 |
| request headers/body | 是 |
| response status/headers | 收到响应时 |
| raw response body | 收到响应数据时 |
| terminal/EOF/close/error metadata | 是 |
| retry/fallback outcome | 是 |

## raw capture 不完整的判据

- 只有 attempt metadata，没有 response body；
- 只有解析后的 JSON，没有原始 SSE framing；
- 只保存最终成功 attempt，缺少失败或被 retry 的 attempt；
- 只保存 client response，没有对应 Provider response；
- 空 body、非法 JSON、错误 event、terminal 后追加帧但原始字节未保存；
- capture 提前结束（cancellation/timeout/process failure/connection close）且未记录已保存范围。

## SSE 分析规则

在 raw 副本上执行：保留原始字节 hash 与长度；按原始 SSE framing 拆帧（`event:`、`data:`、注释、空行、原始换行）；记录 event name、data 长度、JSON 可解析性与 `type`；检查 terminal event、terminal 后事件、EOF、空帧、非法帧；帧序号与 Gateway protocol error 对齐。禁止用 `json.Marshal`、换行归一化或字段重排后的输出替换 raw artifact。
