// The three wires and every translation between them.
//
// The trial's client calls the entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. Each provider speaks one wire of its own
// (openai, anthropic or gemini). This file translates in every direction:
//
//   request:  the client's body into the provider's wire, passing every
//             field through untouched: system prompt, user text (images
//             included), tools, stop sequences, max_tokens, JSON mode.
//             The provider echoes what it parsed, and the client compares
//             that echo with what it sent, so a field rewritten or dropped
//             loses the request.
//   response: the provider's answer back into the wire the client used,
//             streamed or not as it asked, with the provider's text whole,
//             its usage counts as that wire reports them, and its stop
//             reason mapped (end_turn to stop, max_tokens to length,
//             tool_use to tool_calls, refusal to content_filter).
//
// The two rules that shape the translations:
//   - A JSON-mode request carries response_format on the OpenAI wire and
//     responseMimeType on the Gemini wire; the Anthropic wire cannot carry
//     it (its echo always says json_mode false), so a JSON-mode request
//     from an OpenAI client must never go to an anthropic-wire provider.
//     (An Anthropic client never asks for JSON mode in its wire, so its
//     JSON-requirement requests may use any wire whose model supports it.)
//   - An OpenAI stream needs stream_options.include_usage: the usage chunk
//     the provider sends after the finish chunk is the only place the
//     client's usage check can read the input tokens from.
//
// Streaming translations buffer the whole upstream stream first, then emit
// the client's events in the correct order. The client checks the event
// order and the final text and usage, not the timing, and the answers are
// short, so buffering is both correct and deterministic. Everything here
// is a pure function of its arguments: no clock, no randomness, so the same
// body and the same provider answer give the same bytes on a replay. The
// model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './brief.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The ids the entry's own answers carry (fixed, so a replay matches). */
const OPENAI_ANSWER_ID = 'chatcmpl-agg';
const ANTHROPIC_ANSWER_ID = 'msg_agg';
/** The provider splits streamed text into pieces of this many characters. */
const STREAM_PIECE_CHARS = 13;

/** GET /v1/models on an OpenAI-shaped request: OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request: Anthropic's list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function numOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** Split streamed text into the provider-sized pieces the client concatenates back. */
function piecesOf(text: string): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE_CHARS) out.push(text.slice(at, at + STREAM_PIECE_CHARS));
  return out;
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One SSE frame of a stream's text: its event name (null when unnamed) and its data lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Parse SSE text into frames, normalising CRLF. A frame with no data line is dropped. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function openAIStopToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function geminiStopToOpenAI(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  if (reason === 'RECITATION') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
export function geminiStopToAnthropic(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  if (reason === 'RECITATION') return 'refusal';
  return 'end_turn';
}

// ---------------------------------------------------------------------------
// Request translations
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions request naming `model`, with the usage chunk asked for on a stream. */
export function openAIToOpenAIReq(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** An Anthropic Messages request naming `model`. The wire carries no JSON mode and no usage flag. */
export function anthropicToAnthropicReq(body: Json, model: string): Json {
  return { ...body, model };
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function anthropicBlockToPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') {
      return { type: 'image_url', image_url: { url: source.url } };
    }
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAIReq(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = anthropicBlockToPart(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an image; null for the rest. */
function openAIPartToBlock(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return {
        type: 'image',
        source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] },
      };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that
 * names `model`. The Anthropic wire cannot carry JSON mode, so a caller
 * must never send a JSON-mode request here: the provider would echo
 * json_mode false and the client would fail the echo check.
 */
export function openAIToAnthropicReq(body: Json, model: string): Json {
  const out: Json = { model };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const anthropicMessages: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      anthropicMessages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      anthropicMessages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) anthropicMessages.push({ role: 'assistant', content: blocks });
      continue;
    }
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToBlock(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) anthropicMessages.push({ role: 'user', content: blocks });
  }
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  out.messages = anthropicMessages;
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) {
    out.stop_sequences = body.stop.filter((s): s is string => typeof s === 'string');
  }
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters)
            ? raw.function.parameters
            : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function)) {
    out.tool_choice = { type: 'tool', name: typeof choice.function.name === 'string' ? choice.function.name : '' };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** The media type and base64 data of a data URL, or null when the URL is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
}

/** One OpenAI content part as a Gemini part: text, inline image data, or a file reference. */
function openAIPartToGeminiPart(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = dataUrlParts(url);
    if (data !== null) return { inlineData: { mimeType: data.media, data: data.data } };
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** One Anthropic content block as a Gemini part. */
function anthropicBlockToGeminiPart(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { inlineData: { mimeType: media, data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** The generationConfig of a Gemini request: cap, sampling, stops, JSON mode. */
function geminiConfig(opts: {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stops?: string[];
  jsonMode?: boolean;
}): Json {
  const config: Json = {};
  if (opts.maxTokens !== undefined) config.maxOutputTokens = opts.maxTokens;
  if (opts.temperature !== undefined) config.temperature = opts.temperature;
  if (opts.topP !== undefined) config.topP = opts.topP;
  if (opts.stops !== undefined && opts.stops.length > 0) config.stopSequences = opts.stops;
  if (opts.jsonMode === true) config.responseMimeType = 'application/json';
  return config;
}

function stopStrings(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) return value.filter((s): s is string => typeof s === 'string');
  return [];
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body.
 * The model goes in the URL's path, not the body. System messages become
 * the systemInstruction; each other message becomes a content (user, or
 * model for the assistant); tools become functionDeclarations.
 */
export function openAIToGeminiReq(body: Json, _model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const contents: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') continue; // A tool message carries no echoable text; the call itself is the answer's.
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const gp = openAIPartToGeminiPart(part);
        if (gp !== null) parts.push(gp);
      }
    }
    if (raw.role === 'assistant') {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
    } else if (parts.length > 0) {
      contents.push({ role: 'user', parts });
    }
  }
  const out: Json = {};
  if (systemParts.length > 0) {
    out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  out.generationConfig = geminiConfig({
    maxTokens:
      typeof body.max_tokens === 'number'
        ? body.max_tokens
        : typeof body.max_completion_tokens === 'number'
          ? body.max_completion_tokens
          : undefined,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: stopStrings(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
  });
  return out;
}

/**
 * An Anthropic Messages request as a Gemini generateContent body. The
 * Anthropic wire carries no JSON mode, so no responseMimeType is ever set.
 */
export function anthropicToGeminiReq(body: Json, _model: string): Json {
  const system = textOf(body.system);
  const out: Json = {};
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          parts.push({
            functionResponse: {
              name: '',
              response: { content: toolResultText(block.content) },
            },
          });
          continue;
        }
        if (assistant && block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
          continue;
        }
        const gp = anthropicBlockToGeminiPart(block);
        if (gp !== null) parts.push(gp);
      }
    }
    if (parts.length > 0) contents.push({ role: assistant ? 'model' : 'user', parts });
  }
  out.contents = contents;
  if (Array.isArray(body.tools)) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      ];
    });
    if (decls.length > 0) out.tools = [{ functionDeclarations: decls }];
  }
  out.generationConfig = geminiConfig({
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops: Array.isArray(body.stop_sequences)
      ? body.stop_sequences.filter((s): s is string => typeof s === 'string')
      : [],
  });
  return out;
}

// ---------------------------------------------------------------------------
// Response translations (non-stream)
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIRespToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: openAIStopToAnthropic(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numOrZero(usage.prompt_tokens), output_tokens: numOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicRespToOpenAI(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const textParts: string[] = [];
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
        type: 'function',
        function: {
          name: typeof block.name === 'string' ? block.name : '',
          arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
        },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: textParts.join('') };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: anthropicStopToOpenAI(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numOrZero(usage.input_tokens),
      completion_tokens: numOrZero(usage.output_tokens),
      total_tokens: numOrZero(usage.input_tokens) + numOrZero(usage.output_tokens),
    },
  };
}

/** The text of a Gemini answer: its candidates' parts joined. */
function geminiText(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts
    .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
    .join('');
}

/** A Gemini answer's usage: prompt and candidate token counts. */
function geminiUsage(body: Json): { input: number; output: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { input: numOrZero(meta.promptTokenCount), output: numOrZero(meta.candidatesTokenCount) };
}

/** A Gemini answer's finish reason: its first candidate's. */
function geminiFinish(body: Json): unknown {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return first.finishReason ?? null;
}

/** A Gemini generateContent answer as a Chat Completions object. */
export function geminiRespToOpenAI(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  return {
    id: OPENAI_ANSWER_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: geminiText(body) },
        finish_reason: geminiStopToOpenAI(geminiFinish(body)),
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiRespToAnthropic(body: Json, model: string): Json {
  const usage = geminiUsage(body);
  const text = geminiText(body);
  return {
    id: ANTHROPIC_ANSWER_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: geminiStopToAnthropic(geminiFinish(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// Stream translations: buffer the upstream, then emit the client's events
// in order. The client checks the order and the final text and usage, not
// the timing, and the answers are short.
// ---------------------------------------------------------------------------

/** The text, usage and stop of one upstream answer, read off its SSE frames. */
export interface UpstreamAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** The stop reason in the client's wire already (stop / end_turn). */
  stop: string;
}

/** An OpenAI provider's SSE stream as its text, usage and stop reason. */
export function readOpenAIStream(text: string): UpstreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'stop';
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numOrZero(chunk.usage.prompt_tokens);
      output = numOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** An Anthropic provider's SSE stream as its text, usage and stop reason. */
export function readAnthropicStream(text: string): UpstreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') out += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        out += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage)) output = numOrZero(data.usage.output_tokens);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/** A Gemini provider's SSE stream as its text, usage and stop reason. */
export function readGeminiStream(text: string): UpstreamAnswer {
  let out = '';
  let input = 0;
  let output = 0;
  let stop = 'STOP';
  for (const frame of parseSseFrames(text)) {
    let chunk: unknown;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : null;
    if (first !== null) {
      const content = isRecord(first.content) ? first.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') out += part.text;
      }
      if (typeof first.finishReason === 'string') stop = first.finishReason;
    }
    if (isRecord(chunk.usageMetadata)) {
      input = numOrZero(chunk.usageMetadata.promptTokenCount);
      output = numOrZero(chunk.usageMetadata.candidatesTokenCount);
    }
  }
  return { text: out, inputTokens: input, outputTokens: output, stop };
}

/**
 * One buffered answer as an OpenAI event stream: a role chunk, the text in
 * pieces, one finish chunk, the usage chunk, then [DONE] last. The usage
 * chunk carries both counts, which the client's usage check reads.
 */
export function openAIStreamEvents(answer: UpstreamAnswer, model: string): string[] {
  const events: string[] = [];
  const base = { id: OPENAI_ANSWER_ID, object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model };
  events.push(
    sseEvent(null, { ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }),
  );
  for (const piece of piecesOf(answer.text)) {
    events.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] }));
  }
  events.push(sseEvent(null, { ...base, choices: [{ index: 0, delta: {}, finish_reason: answer.stop }] }));
  events.push(
    sseEvent(null, {
      ...base,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  events.push('data: [DONE]\n\n');
  return events;
}

/**
 * One buffered answer as an Anthropic event stream: message_start, one
 * text block opened, its deltas, the block closed, message_delta with the
 * stop reason and the output tokens, then message_stop. Only the output
 * count is checked on this wire, but the input count is carried too.
 */
export function anthropicStreamEvents(answer: UpstreamAnswer, model: string): string[] {
  const events: string[] = [];
  events.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: ANTHROPIC_ANSWER_ID,
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  );
  events.push(
    sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of piecesOf(answer.text)) {
    events.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      }),
    );
  }
  events.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: answer.stop, stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
  );
  events.push(sseEvent('message_stop', { type: 'message_stop' }));
  return events;
}
