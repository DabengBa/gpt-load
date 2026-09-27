// Registers the @shared/* alias resolver for node --test runs.
import { register } from 'node:module'

register(new URL('./shared-alias-hooks.mjs', import.meta.url))
