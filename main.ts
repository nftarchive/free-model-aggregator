// free-model-aggregator-01: serve every request with the best free model you can get.
//
// One long-running server. It reads the brief (provider directory, quality
// table, workload manifest), probes every provider's catalog route, builds
// one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and
//      POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//      A process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document.
//
// ROUTING (the whole point)
//   For each request, choose the strongest free model that meets its
//   requirements and still has budget; speak that provider's wire; when a
//   provider refuses, count the failure and answer the request with
//   another; return the answer in the wire the client used. The plan is
//   made once at startup from the manifest, the quality table and the
//   probes (see lib/router.ts): a greedy assignment in manifest order with
//   a load penalty that spreads the trial over providers, so no one
//   provider's free-tier budget runs out mid-trial. At runtime each
//   request tries its planned primary first; a refusal fails over to the
//   next best candidate by the same score against the calls actually made.
//
// THE WIRES (see lib/translate.ts)
//   The client speaks OpenAI or Anthropic; each provider speaks one of
//   openai, anthropic or gemini. Every direction is translated: request
//   fields pass through untouched (system, user text with images, tools,
//   stops, max_tokens, JSON mode), and the answer comes back in the
//   client's wire with its text whole, its usage counts and its stop
//   reason mapped. A JSON-mode request from an OpenAI client never goes
//   to an anthropic-wire provider (that wire cannot carry JSON mode).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  joinBase,
  modelFree,
  probeProvider,
  type CatalogReport,
  type Probe,
} from './lib/catalog.ts';
import {
  isRecord,
  ratingOfModel,
  readBrief,
  type Brief,
  type Json,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicRespToOpenAI,
  anthropicStreamEvents,
  anthropicToAnthropicReq,
  anthropicToGeminiReq,
  anthropicToOpenAIReq,
  geminiRespToAnthropic,
  geminiRespToOpenAI,
  openAIError,
  openAIModelList,
  openAIRespToAnthropic,
  openAIStreamEvents,
  openAIToAnthropicReq,
  openAIToGeminiReq,
  openAIToOpenAIReq,
  readAnthropicStream,
  readGeminiStream,
  readOpenAIStream,
  type ServedModel,
  type UpstreamAnswer,
} from './lib/translate.ts';
import {
  candidatesFor,
  planAssignments,
  tryOrder,
  type Candidate,
} from './lib/router.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The planned primary (provider, model) per manifest request id. */
let planned = new Map<string, Candidate>();
/** How many provider calls have been made, per provider id (failed calls count: they spend budget). */
const callsByProvider = new Map<string, number>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider, build the one catalog, and plan the manifest's primaries. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)),
  );
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  const manifest = brief.workload?.manifest ?? [];
  // A JSON-mode request from an OpenAI client never goes to an anthropic-wire provider; the
  // manifest's api tells which wire the client will use for each request.
  const clientJsonById = new Map<string, boolean>();
  for (const request of manifest) {
    clientJsonById.set(request.id, request.api === 'openai' && request.requires.json === true);
  }
  planned = planAssignments(manifest, probes, brief.ratings, clientJsonById);
  log(
    `planned ${planned.size} of ${manifest.length} requests over ${probes.filter((p) => p.ok).length} providers`,
  );
}

/** The free models across the providers that answered, in probe order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Whether the client asked for JSON mode on the OpenAI wire. */
function clientJsonMode(clientWire: 'openai' | 'anthropic', body: Json): boolean {
  if (clientWire !== 'openai') return false;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

/**
 * The requirements of the request now arriving, and its manifest id when
 * the manifest has a row for it. The manifest is the declared source; the
 * body's own fields are the fallback for a brief that holds none.
 */
function requirementsFor(body: Json): { requires: Requirement; requestId: string | undefined } {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, requestId: entry.id };
  return { requires: requirementsFromBody(body), requestId: undefined };
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The provider's infer URL for one candidate: the model goes in the path on the gemini wire. */
function providerUrl(candidate: Candidate, stream: boolean): string {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  return `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`;
}

/** The client's body translated into one candidate provider's wire. */
function providerBody(clientWire: 'openai' | 'anthropic', candidate: Candidate, body: Json): Json {
  const wire = candidate.probe.provider.wire;
  if (clientWire === 'openai') {
    if (wire === 'openai') return openAIToOpenAIReq(body, candidate.model.id);
    if (wire === 'anthropic') return openAIToAnthropicReq(body, candidate.model.id);
    return openAIToGeminiReq(body, candidate.model.id);
  }
  if (wire === 'openai') return anthropicToOpenAIReq(body, candidate.model.id);
  if (wire === 'anthropic') return anthropicToAnthropicReq(body, candidate.model.id);
  return anthropicToGeminiReq(body, candidate.model.id);
}

/** One upstream answer's stop reason, mapped into the client's wire. */
function clientStop(answer: UpstreamAnswer, providerWire: string, clientWire: 'openai' | 'anthropic'): string {
  if (clientWire === 'openai') {
    if (providerWire === 'anthropic') {
      return answer.stop === 'max_tokens'
        ? 'length'
        : answer.stop === 'tool_use'
          ? 'tool_calls'
          : answer.stop === 'refusal'
            ? 'content_filter'
            : 'stop';
    }
    if (providerWire === 'gemini') {
      return answer.stop === 'MAX_TOKENS'
        ? 'length'
        : answer.stop === 'SAFETY' || answer.stop === 'RECITATION'
          ? 'content_filter'
          : 'stop';
    }
    return answer.stop;
  }
  if (providerWire === 'openai') {
    return answer.stop === 'length'
      ? 'max_tokens'
      : answer.stop === 'tool_calls' || answer.stop === 'function_call'
        ? 'tool_use'
        : answer.stop === 'content_filter'
          ? 'refusal'
          : 'end_turn';
  }
  if (providerWire === 'gemini') {
    return answer.stop === 'MAX_TOKENS'
      ? 'max_tokens'
      : answer.stop === 'SAFETY' || answer.stop === 'RECITATION'
        ? 'refusal'
        : 'end_turn';
  }
  return answer.stop;
}

/**
 * Serve one client request: try its planned primary first, then the next
 * best candidates by score, until one provider answers. A refusal (402,
 * 429, 5xx, network error) counts the call and moves on; the first
 * success ends the request. Returns the client's answer (JSON object or
 * SSE event strings), or null when no provider answered.
 */
async function serve(
  clientWire: 'openai' | 'anthropic',
  body: Json,
  requires: Requirement,
  requestId: string | undefined,
): Promise<{ json: Json } | { events: string[] } | null> {
  const stream = body.stream === true;
  const jsonMode = clientJsonMode(clientWire, body);
  const candidates = candidatesFor(probes, brief.ratings, requires, jsonMode);
  if (candidates.length === 0) return null;

  // The planned primary first when it is still a candidate; the rest by
  // score against the calls actually made so far.
  const order: Candidate[] = [];
  const seen = new Set<string>();
  const primary = requestId !== undefined ? planned.get(requestId) : undefined;
  if (
    primary !== undefined &&
    candidates.some((c) => c.probe.provider.id === primary.probe.provider.id && c.model.id === primary.model.id)
  ) {
    order.push(primary);
    seen.add(`${primary.probe.provider.id}/${primary.model.id}`);
  }
  for (const candidate of tryOrder(candidates, callsByProvider, new Set())) {
    const key = `${candidate.probe.provider.id}/${candidate.model.id}`;
    if (!seen.has(key)) {
      order.push(candidate);
      seen.add(key);
    }
  }

  const failedProviders = new Set<string>();
  for (const candidate of order) {
    if (failedProviders.has(candidate.probe.provider.id)) continue;
    const provider = candidate.probe.provider;
    const url = providerUrl(candidate, stream);
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      ...authHeaders(provider.auth, provider.key),
    };
    if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
    if (stream) headers.accept = 'text/event-stream';
    let upstream: Response;
    try {
      upstream = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(providerBody(clientWire, candidate, body)),
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      // The provider is unreachable: count the attempt and try the next one.
      callsByProvider.set(provider.id, (callsByProvider.get(provider.id) ?? 0) + 1);
      failedProviders.add(provider.id);
      log(`call to ${provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
      continue;
    }
    callsByProvider.set(provider.id, (callsByProvider.get(provider.id) ?? 0) + 1);
    if (!upstream.ok) {
      // A refusal is not an answer: count it and answer with another model.
      failedProviders.add(provider.id);
      const detail = await upstream.text().catch(() => '');
      log(`call to ${provider.id}/${candidate.model.id} refused HTTP ${upstream.status}: ${detail.slice(0, 200)}`);
      continue;
    }
    // One success ends the request.
    if (stream) {
      const text = await upstream.text();
      const answer =
        provider.wire === 'openai'
          ? readOpenAIStream(text)
          : provider.wire === 'anthropic'
            ? readAnthropicStream(text)
            : readGeminiStream(text);
      answer.stop = clientStop(answer, provider.wire, clientWire);
      const events =
        clientWire === 'openai'
          ? openAIStreamEvents(answer, candidate.model.id)
          : anthropicStreamEvents(answer, candidate.model.id);
      return { events };
    }
    const answer: unknown = await upstream.json();
    if (!isRecord(answer)) return null;
    if (clientWire === 'openai') {
      if (provider.wire === 'openai') return { json: answer };
      if (provider.wire === 'anthropic') return { json: anthropicRespToOpenAI(answer, candidate.model.id) };
      return { json: geminiRespToOpenAI(answer, candidate.model.id) };
    }
    if (provider.wire === 'openai') return { json: openAIRespToAnthropic(answer, candidate.model.id) };
    if (provider.wire === 'anthropic') return { json: answer };
    return { json: geminiRespToAnthropic(answer, candidate.model.id) };
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const { requires, requestId } = requirementsFor(body);
  const served = await serve('openai', body, requires, requestId);
  if (served === null) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  }
  if ('events' in served) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const event of served.events) res.write(event);
    res.end();
    return;
  }
  sendJson(res, 200, served.json);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const { requires, requestId } = requirementsFor(body);
  const served = await serve('anthropic', body, requires, requestId);
  if (served === null) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  }
  if ('events' in served) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for (const event of served.events) res.write(event);
    res.end();
    return;
  }
  sendJson(res, 200, served.json);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
