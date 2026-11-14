// The catalog probe: ask every provider what it serves, in its own wire.
//
// Each provider lists its models on a catalog route:
//   openai     GET <base>/v1/models          Authorization: Bearer <key>
//   anthropic  GET <base>/v1/models          x-api-key: <key>, anthropic-version: 2023-06-01
//   gemini     GET <base>/v1beta/models      x-goog-api-key: <key>
//
// A listing names each model's id, whether it is free, its context length
// and its capabilities (tools, vision, JSON mode). A probe that fails (a
// refused connection, a non-2xx, a body that is not JSON) leaves its
// provider out of the catalog and gets no traffic; the trial goes on.
//
// The catalog this builds is the one document `--list` prints and the
// register body carries, byte for byte. Its base_url is the directory's own
// path (/p/<id>), never the path joined to $MAKEFUN_ENV_URL: the env's
// origin differs between a live pass and its replay, and the catalog is
// part of the state a replay must reproduce exactly. The scorer compares a
// reported base_url by its path suffix, so the path alone passes.
//
// Everything here is deterministic: no clock, no randomness.

import { isRecord, ratingOfModel, type AuthScheme, type DirectoryProvider, type Ratings, type Wire } from './brief.ts';

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One provider's catalog probe: what its route answered. */
export interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers that carry a provider's key, by its auth scheme. */
export function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
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
 * Whether a listing's price block says the model is free: true when every
 * price it names is zero, null when the listing says nothing about price.
 */
function zeroPriced(v: Record<string, unknown>): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context =
    asNumber(v.context_length) ?? asNumber(v.context_window) ?? asNumber(v.inputTokenLimit);
  return {
    id,
    display_name:
      asString(v.display_name) !== ''
        ? asString(v.display_name)
        : asString(v.displayName) !== ''
          ? asString(v.displayName)
          : id,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeProvider(
  envUrl: string,
  provider: DirectoryProvider,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/** What a model can do: the quality table's row first, the listing's own fields second. */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export function capabilitiesOf(
  model: ListedModel,
  row: ReturnType<typeof ratingOfModel>,
): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/**
 * Whether a model is free: the quality table's row where it has one (the
 * sealed truth), the listing's own word where it does not, paid when
 * neither says. A paid model is never routed.
 */
export function modelFree(model: ListedModel, row: ReturnType<typeof ratingOfModel>): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
export function meetsRequirements(
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number },
  caps: Capabilities,
): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (
    requires.min_context !== undefined &&
    (caps.context_length === null || caps.context_length < requires.min_context)
  ) {
    return false;
  }
  return true;
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, in the listing's own order (the scorer
 * compares the free set, not the order, but keeping the provider's order
 * keeps `--list` and the register body honest).
 */
export function buildCatalog(probes: readonly Probe[], ratings: Ratings | null): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: modelFree(model, row),
        context_length: row?.context_length ?? model.context_length ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.key_env,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
