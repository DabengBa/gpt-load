import { useMemo, useState } from 'react'

import {
  channelIconRasterURL,
  namespacedChannelIconMarkup,
  nextChannelIconInstanceId,
} from '@shared/assets/channel-icons'

import { channelIconImage } from './channel-icon-image'
import './ChannelIcon.css'

// Always pass channel-definition metadata. Mapping a ChannelID to an asset
// belongs to the channel definition/compiler, never to an individual view.
export function ChannelIcon({ icon, mark }: { icon: string; mark: string }) {
  const [instanceId] = useState(() => nextChannelIconInstanceId())
  const image = useMemo(() => {
    const markup = namespacedChannelIconMarkup(icon, instanceId)
    return markup === null ? null : channelIconImage(markup)
  }, [icon, instanceId])
  const rasterURL = channelIconRasterURL(icon)

  if (image !== null) {
    if (image.mode === 'mask') {
      return (
        <span
          className="channel-icon channel-icon--mask"
          aria-hidden="true"
          style={{ maskImage: `url("${image.src}")` }}
        />
      )
    }
    return (
      <span className="channel-icon" aria-hidden="true">
        <img src={image.src} alt="" />
      </span>
    )
  }
  if (rasterURL !== null) {
    return (
      <span className="channel-icon channel-icon--raster" aria-hidden="true">
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
