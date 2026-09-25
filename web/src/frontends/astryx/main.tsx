import * as stylex from '@stylexjs/stylex'
import { createRoot } from 'react-dom/client'

import './entry.css'

const styles = stylex.create({
  shell: {
    display: 'grid',
    minHeight: '100vh',
    placeItems: 'center',
  },
})

function Shell() {
  return <main {...stylex.props(styles.shell)}>GPT-Load</main>
}

const host = document.getElementById('app')
if (!host) throw new Error('missing #app mount point')
createRoot(host).render(<Shell />)
