# Group Site Information Request

## Original statement

User request, 2026-11-06:

> 思考功能:
> 在维护分组的时候, 增加一个'访问令牌'
> 用来通过这个令牌, 访问newapi站点,获取"钱包余额\站点公告\api key分组倍率"信息.

## Later confirmed scope

The user later confirmed the following scope in the same conversation. This is
a normalized record of the confirmations, not a verbatim quotation: do not add
a separate provider type; derive the site root from the provider website; add
a site user ID; refresh automatically every 60 minutes; show balance and ratio
in the group collection; provide a separate announcements page ordered by
event time. Access tokens are encrypted and not echoed, are omitted when
copying a group, and a changed site root requires confirmation before sending
a token.

## Interpretation boundary

The original statement requests access to wallet balance, announcements and
API Key group ratio. The later confirmed scope above is recorded separately
from the exact original statement. Product contracts are owned by
[`feature.group-site-information`](../db/features/group-site-information.md).
