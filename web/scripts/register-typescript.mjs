import { readFileSync } from 'node:fs'
import { registerHooks } from 'node:module'
import ts from 'typescript'

registerHooks({
  load(url, context, nextLoad) {
    if (!url.endsWith('.ts')) return nextLoad(url, context)

    const source = readFileSync(new URL(url), 'utf8')
    const output = ts.transpileModule(source, {
      compilerOptions: {
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
      fileName: new URL(url).pathname,
    })

    return { format: 'module', source: output.outputText, shortCircuit: true }
  },
})
