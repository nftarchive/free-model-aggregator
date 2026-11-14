// The trial brief: three JSON documents carried in one text file.
//
// A trial hands this server a brief (the path in $MAKEFUN_PROMPT_FILE) whose
// prose wraps three documents, found by their `schema` member:
//
//   - the provider directory (free-model-aggregator-01/providers/v1): the ten
//     simulated free-tier providers, each with id, wire (openai, anthropic or
//     gemini), auth scheme, the key the service takes, and the base path to
//     join to $MAKEFUN_ENV_URL.
//   - the quality table (free-model-aggregator-01/ratings/v1): every model's
//     public rating in milli-points, what it can do (tools, vision, JSON
//     mode, context length), whether it is free, and each request's best
//     legal rating. The scorer ranks an answer by this rating, so routing
//     reads it directly.
//   - the workload manifest (free-model-aggregator-01/workload/v1): the
//     twenty requests the client will send, in order, each with its api
//     (openai or anthropic), whether it streams, its weight, and the
//     requirements a model must meet to serve it.
//
// Everything here is a pure function of the file's text: no clock, no
// randomness, so the same brief always parses to the same structure.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

/** A JSON object, the only shape this file works with. */
export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. */
export interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  /** The base path (or absolute URL) the provider's routes hang off. */
  base: string;
  key_env: string;
  key: string;
}

export interface Directory {
  schema: string;
  providers: DirectoryProvider[];
}

/** One row of the quality table: a model's public rating and capabilities. */
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Ratings {
  schema: string;
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
}

/** What one request needs from its model. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

export interface Workload {
  schema: string;
  manifest: WorkloadRequest[];
}

/** The parsed brief. `error` is set when the file could not be read at all. */
export interface Brief {
  directory: Directory;
  ratings: Ratings | null;
  workload: Workload | null;
  error: string | null;
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBool(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/**
 * Every JSON object in `text` that carries a string `schema`, in the order
 * they appear. The scanner walks `{` characters and, for each, finds the
 * matching close brace while skipping string contents, then tries to parse
 * just that slice. A slice that is not an object, or not one with a schema,
 * is skipped and the scan moves on.
 */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    let depth = 0;
    let inString = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inString) {
        if (ch === '\\') i += 1;
        else if (ch === '"') inString = false;
      } else if (ch === '"') inString = true;
      else if (ch === '{') depth += 1;
      else if (ch === '}') {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not a standalone object: keep scanning.
    }
  }
  return out;
}

function parseProvider(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The directory writes `path` (/p/<id>); an absolute base may also be
  // spelled base_path, base_url or base. Take the first non-empty, path first.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return {
    id,
    name: asString(v.name) !== '' ? asString(v.name) : id,
    wire,
    auth,
    base,
    key_env: asString(v.key_env),
    key: asString(v.key),
  };
}

function parseRating(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asNumber(v.rating_milli) ?? 0,
    context_length: asNumber(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function parseWorkloadRequest(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: req.tools === true,
      vision: req.vision === true,
      json: req.json === true,
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
        ? { min_context: req.min_context }
        : {}),
    },
  };
}

/** Read and parse the brief at `path`. */
export function readBrief(path: string): Brief {
  const directory: Directory = { schema: PROVIDERS_SCHEMA, providers: [] };
  if (path === '') {
    return { directory, ratings: null, workload: null, error: 'MAKEFUN_PROMPT_FILE is not set' };
  }
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory, ratings: null, workload: null, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { directory, ratings: null, workload: null, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = parseProvider(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: Ratings | null =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? null
      : {
          schema: RATINGS_SCHEMA,
          ratings: ratingsDoc.ratings.flatMap((v) => {
            const r = parseRating(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isRecord(v) || asString(v.id) === '') return [];
                return [{ id: asString(v.id), rating_milli: asNumber(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: Workload | null =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? null
      : {
          schema: WORKLOAD_SCHEMA,
          manifest: workloadDoc.manifest.flatMap((v) => {
            const w = parseWorkloadRequest(v);
            return w === null ? [] : [w];
          }),
        };
  return { directory: { schema: PROVIDERS_SCHEMA, providers }, ratings, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingOfModel(ratings: Ratings | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}
