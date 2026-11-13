import assert from 'node:assert/strict'
import test from 'node:test'

import type { ModelRouteScheduleDetailDto } from '../src/shared/control/resources/model-route-schedule'
import { scheduleRows } from '../src/frontends/astryx/features/monitor/schedule-detail-model.ts'

test('saved priority tiers sort by exact configured share, with disabled rows at zero and stable ties', () => {
  const candidate = (
    entry_id: string,
    priority: number,
    configured_share: number,
    enabled = true,
  ) => ({
    entry_id,
    priority,
    configured_share,
    enabled,
  })
  const detail = {
    groups: [
      { group_id: 2, enabled: true, entries: [candidate('tie-b', 1, 0.2)] },
      {
        group_id: 1,
        enabled: true,
        entries: [
          candidate('low', 1, 0.19999),
          candidate('disabled', 1, 0.9, false),
          candidate('tie-a', 1, 0.2),
          candidate('high', 1, 0.20001),
          candidate('next-tier', 2, 1),
        ],
      },
      { group_id: 3, enabled: false, entries: [candidate('disabled-group', 1, 1)] },
    ],
  } as unknown as ModelRouteScheduleDetailDto
  assert.deepEqual(
    scheduleRows(detail).map(({ entry }) => entry.entry_id),
    ['high', 'tie-a', 'tie-b', 'low', 'disabled', 'disabled-group', 'next-tier'],
  )
})
