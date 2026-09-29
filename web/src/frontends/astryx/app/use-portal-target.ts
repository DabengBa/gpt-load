import { useEffect, useState } from 'react'

/**
 * Resolves a document-level portal anchor by id — the React counterpart of
 * classic `<Teleport defer to="#id">`. Returns null until the anchor exists,
 * so callers render `createPortal(node, target)` conditionally. Re-resolves
 * whenever `id` changes; callers whose anchor mounts in the same commit get
 * the element on the follow-up render (effects run after the whole tree is
 * in the DOM).
 */
export function usePortalTarget(id: string): HTMLElement | null {
  const [target, setTarget] = useState<HTMLElement | null>(null)
  useEffect(() => {
    setTarget(document.getElementById(id))
  }, [id])
  return target
}
