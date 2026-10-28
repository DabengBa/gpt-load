// Registers the @shared/* alias resolver for node --test runs.
import { registerHooks } from 'node:module'

registerHooks(await import('./shared-alias-hooks.mjs'))
