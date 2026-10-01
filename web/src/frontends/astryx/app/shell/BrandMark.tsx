import * as stylex from '@stylexjs/stylex'

const styles = stylex.create({
  mark: {
    display: 'block',
    flexShrink: 0,
    color: 'var(--color-action)',
  },
})

export function BrandMark({ size = 24, label }: { size?: number | string; label?: string }) {
  const dimension = typeof size === 'number' ? String(size) : size
  return (
    <svg
      {...stylex.props(styles.mark)}
      viewBox="0 0 24 24"
      width={dimension}
      height={dimension}
      fill="currentColor"
      role={label ? 'img' : undefined}
      aria-hidden={label ? undefined : 'true'}
      aria-label={label}
    >
      <g transform="rotate(45 12 12)">
        <path d="M4 6.6H8V16H14.4V20H4Z" />
        <path d="M20 17.4H16V8H9.6V4H20Z" />
      </g>
    </svg>
  )
}
