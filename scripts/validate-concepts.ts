import { glob } from 'glob'
import { readFileSync } from 'fs'
import path from 'path'

const REQUIRED_FIELDS = ['id', 'title', 'subject', 'description', 'sliderParams', 'Scene']
const VALID_SUBJECTS = [
  'electromagnetics',
  'circuits',
  'electronics',
  'power-systems',
  'machines',
  'control',
  'installations',
]

async function main() {
  const files = await glob('src/concepts/*/index.tsx', { cwd: process.cwd() })

  if (files.length === 0) {
    console.log('⚠️  No concepts found.')
    return
  }

  let hasError = false
  const seenIds = new Map<string, string>()

  for (const file of files) {
    const raw = readFileSync(file, 'utf-8')
    const errors: string[] = []

    // Strip comments before checking (only `//` at line start or after whitespace, to spare `//` inside strings)
    const src = raw
      .replace(/(^|\s)\/\/.*$/gm, '$1')
      .replace(/\/\*[\s\S]*?\*\//g, '')

    for (const field of REQUIRED_FIELDS) {
      if (!src.includes(`${field}:`)) {
        errors.push(`Missing field: ${field}`)
      }
    }

    const subjectMatch = src.match(/subject:\s*['"]([^'"]+)['"]/)
    if (subjectMatch && !VALID_SUBJECTS.includes(subjectMatch[1])) {
      errors.push(`Invalid subject: "${subjectMatch[1]}"`)
    }

    const idMatch = src.match(/id:\s*['"]([^'"]+)['"]/)
    if (idMatch) {
      const id = idMatch[1]
      const folderName = path.basename(path.dirname(file))

      if (!/^[a-z0-9-]+$/.test(id)) {
        errors.push(`id must be kebab-case: "${id}"`)
      }
      if (id !== folderName) {
        errors.push(`id "${id}" must match folder name "${folderName}"`)
      }
      if (seenIds.has(id)) {
        errors.push(`duplicate id "${id}" (also in ${seenIds.get(id)})`)
      } else {
        seenIds.set(id, file)
      }
    }

    if (/['"]#[0-9a-fA-F]{3,8}['"]/.test(src)) {
      errors.push('Hard-coded hex color: use usePhysicsColors() (see DESIGN.md)')
    }

    // A formula in a narration step is marked with `backticks`; en and ko must mark the same ones
    for (const [, body] of src.matchAll(/^      text: \{\n(.*?)^      \},$/gms)) {
      const counts = [...body.matchAll(/^\s+(en|ko): '(.*)',$/gm)].map(
        ([, loc, line]) => [loc, (line.match(/`/g) ?? []).length] as const
      )
      for (const [loc, n] of counts) {
        if (n % 2 !== 0) errors.push(`story step: unclosed \` in ${loc}: ${body.slice(0, 60)}`)
      }
      if (counts.length === 2 && counts[0][1] !== counts[1][1]) {
        errors.push(`story step: en marks ${counts[0][1] / 2} formula(s), ko marks ${counts[1][1] / 2}`)
      }
    }

    if (/sliderParams\s*:\s*\[\s*\]/s.test(src)) {
      errors.push('sliderParams must have at least 1 entry')
    }

    if (!/^[\s]*export\s+default\b/m.test(src)) {
      errors.push('Missing: export default concept')
    }

    if (errors.length > 0) {
      console.error(`❌ ${file}:\n  ${errors.join('\n  ')}`)
      hasError = true
    } else {
      console.log(`✅ ${file}`)
    }
  }

  if (hasError) {
    console.error('\n🚫 Validation failed.')
    process.exit(1)
  }
  console.log(`\n✅ All ${files.length} concept(s) valid.`)
}

main()
