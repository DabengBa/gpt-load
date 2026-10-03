export interface ChannelIconImage {
  readonly mode: 'img' | 'mask'
  readonly src: string
}

// Masks preserve inherited color; images preserve brand colors and gradients.
export function channelIconImage(markup: string): ChannelIconImage {
  const src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`
  return { mode: markup.includes('currentColor') ? 'mask' : 'img', src }
}
