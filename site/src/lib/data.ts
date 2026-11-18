// Build-time data: the notebook's results/*.csv and its code cells, read straight from ../beginner.
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const BEGINNER = resolve(process.cwd(), '../beginner');

export type Row = Record<string, number | string>;

export function csv(name: string): Row[] {
  const path = resolve(BEGINNER, 'results', name);
  if (!existsSync(path)) return [];
  const [head, ...lines] = readFileSync(path, 'utf8').trim().split('\n');
  const keys = splitLine(head);
  return lines.map((line) => {
    const cells = splitLine(line);
    return Object.fromEntries(keys.map((k, i) => [k, toValue(cells[i] ?? '')]));
  });
}

function splitLine(line: string): string[] {
  const out: string[] = [];
  let cur = '', quoted = false;
  for (const ch of line) {
    if (ch === '"') quoted = !quoted;
    else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

function toValue(s: string): number | string {
  const n = Number(s);
  return s !== '' && Number.isFinite(n) ? n : s;
}

type Cell = { cell_type: string; source: string | string[] };
const cells: Cell[] = JSON.parse(readFileSync(resolve(BEGINNER, 'submission.ipynb'), 'utf8')).cells;

/** Source of the first code cell starting with `prefix` (keeps the site in sync with the notebook). */
export function code(prefix: string): string {
  for (const c of cells) {
    const src = Array.isArray(c.source) ? c.source.join('') : c.source;
    if (c.cell_type === 'code' && src.startsWith(prefix)) return src;
  }
  return `# (cell starting with "${prefix}" not found in submission.ipynb)`;
}

export const num = (r: Row, k: string) => Number(r[k]);
