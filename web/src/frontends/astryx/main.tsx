import { Button } from '@astryxdesign/core/Button'
import { Theme } from '@astryxdesign/core/theme'
import * as stylex from '@stylexjs/stylex'
import { createRoot } from 'react-dom/client'

import './entry.css'
import { gptloadTheme } from './theme/gptload'
import { useThemePreference } from './theme/theme-preference'

const styles = stylex.create({
  shell: {
    display: 'grid',
    minHeight: '100vh',
    placeItems: 'center',
    backgroundColor: 'var(--color-background-body)',
    color: 'var(--color-text-primary)',
    fontFamily: 'var(--font-family-body)',
    fontSize: 'var(--text-body-size)',
  },
  row: {
    display: 'flex',
    gap: '8px',
  },
  overrideProbe: {
    borderRadius: '2px',
  },
})

function Shell() {
  const [mode] = useThemePreference()
  return (
    <Theme theme={gptloadTheme} mode={mode}>
      <main {...stylex.props(styles.shell)}>
        <div {...stylex.props(styles.row)}>
          <Button label="GPT-Load" />
          <Button label="Override" xstyle={styles.overrideProbe} />
        </div>
      </main>
    </Theme>
  )
}

const host = document.getElementById('app')
if (!host) throw new Error('missing #app mount point')
createRoot(host).render(<Shell />)
