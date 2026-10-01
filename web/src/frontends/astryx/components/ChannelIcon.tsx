import { useMemo, useState } from 'react'

import {
  channelIconRasterURL,
  namespacedChannelIconMarkup,
  nextChannelIconInstanceId,
} from '@shared/assets/channel-icons'

import './ChannelIcon.css'

// Always pass channel-definition metadata. Mapping a ChannelID to an asset
// belongs to the channel definition/compiler, never to an individual view.
export function ChannelIcon({ icon, mark }: { icon: string; mark: string }) {
  const [instanceId] = useState(() => nextChannelIconInstanceId())
  const markup = useMemo(() => namespacedChannelIconMarkup(icon, instanceId), [icon, instanceId])
  const rasterURL = channelIconRasterURL(icon)

  if (markup !== null) {
    // The vendored build-time SVG markup is ours, not user input — same
    // contract as the Vue v-html render in ChannelIcon.vue.
    return (
      <span
        className="channel-icon"
        aria-hidden="true"
        dangerouslySetInnerHTML={{ __html: markup }}
      />
    )
  }
  if (rasterURL !== null) {
    return (
      <span className="channel-icon" aria-hidden="true">
        <img src={rasterURL} alt="" />
      </span>
    )
  }
  return (
    <span className="channel-icon channel-icon--fallback" aria-hidden="true">
      {mark}
    </span>
  )
}
