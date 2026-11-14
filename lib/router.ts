// Routing: which free model serves each request, and what to try next
// when a provider refuses.
//
// The workload is known in advance (the brief's manifest), the quality
// table ranks every free model, and the probes say which providers
// answered. Routing is a plan over those three facts, plus what the trial
// has taught the router so far (calls made, providers that refused):
//
//   - Candidates for a request are the free models that meet its
//     requirements, ranked by rating. A JSON-mode request from an OpenAI
//     client may only use an OpenAI-wire or Gemini-wire provider: the
//     Anthropic wire cannot carry JSON mode, so its echo would say
//     json_mode false and the client would fail the echo check. (An
//     Anthropic client never asks for JSON mode in its wire, so its
//     JSON-requirement requests may use any wire whose model supports it.)
//   - The primary assignment is greedy in manifest order, with a load
//     penalty: a candidate's score is its rating minus LOAD_PENALTY for
//     every call already planned on its provider. The penalty is what
//     spreads the trial over providers instead of piling every request
//     onto the one best model (whose free-tier budget would run out
//     mid-trial and take the rest of the requests down with it). The
//     penalty is small next to the rating gaps that matter (a weight-2
//     request still takes the best model), and large enough that a
//     provider holding several planned calls yields to a fresh provider
//     whose model is only a few milli-points weaker.
//   - Failover: when a provider refuses (402, 429, 5xx, or a network
//     error), the request is retried on the next best candidate, ranked
//     the same way against the calls actually made so far, skipping the
//     provider that just refused it. The first success ends the request:
//     exactly one successful provider call per request is what the
//     client's one_provider check wants. A provider that refused is not
//     blacklisted for later requests — a failure can be its scheduled
//     one-off refusal, not an exhausted budget — but its call count rose,
//     so the penalty already pushes later requests elsewhere.
//
// Budgets are drawn per trial and never published, so the router never
// assumes one: it spreads the load and adapts to refusals. Everything
// here is a pure function of the brief, the probes and the calls made so
// far: no clock, no randomness, so a replay reproduces the plan.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  type Probe,
  type ListedModel,
} from './catalog.ts';
import { ratingOfModel, type Ratings, type Requirement, type WorkloadRequest } from './brief.ts';

/** One (provider, model) pair that can serve a request, with its rating. */
export interface Candidate {
  probe: Probe;
  model: ListedModel;
  rating: number;
}

/** How many milli-points one planned call on a provider costs its next candidate. */
export const LOAD_PENALTY = 1;

/**
 * Every free model that can serve a request, best rating first. `clientJson`
 * is true when the client itself asked for JSON mode on the OpenAI wire;
 * those requests never go to an anthropic-wire provider.
 */
export function candidatesFor(
  probes: readonly Probe[],
  ratings: Ratings | null,
  requires: Requirement,
  clientJson: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientJson && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** A candidate's score against the calls already planned or made on its provider. */
function scoreOf(candidate: Candidate, callsByProvider: ReadonlyMap<string, number>): number {
  return candidate.rating - LOAD_PENALTY * (callsByProvider.get(candidate.probe.provider.id) ?? 0);
}

/** The best candidate by score (rating minus load penalty), or null when there is none. */
export function bestByScore(
  candidates: readonly Candidate[],
  callsByProvider: ReadonlyMap<string, number>,
): Candidate | null {
  let best: Candidate | null = null;
  let bestScore = -Infinity;
  for (const candidate of candidates) {
    const score = scoreOf(candidate, callsByProvider);
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return best;
}

/**
 * The primary (provider, model) for every manifest request, planned once
 * at startup in manifest order. The plan is a function of the brief and
 * the probes only, so a replay reproduces it.
 */
export function planAssignments(
  manifest: readonly WorkloadRequest[],
  probes: readonly Probe[],
  ratings: Ratings | null,
  clientJsonById: ReadonlyMap<string, boolean>,
): Map<string, Candidate> {
  const planned = new Map<string, number>();
  const assignment = new Map<string, Candidate>();
  for (const request of manifest) {
    const candidates = candidatesFor(
      probes,
      ratings,
      request.requires,
      clientJsonById.get(request.id) ?? false,
    );
    const pick = bestByScore(candidates, planned);
    if (pick === null) continue;
    assignment.set(request.id, pick);
    planned.set(pick.probe.provider.id, (planned.get(pick.probe.provider.id) ?? 0) + 1);
  }
  return assignment;
}

/**
 * The order to try candidates in for one request at runtime: best score
 * first, skipping every provider in `excluded`. The scores read the calls
 * actually made so far, so a provider that has already burned budget or
 * refused falls behind a fresh one.
 */
export function tryOrder(
  candidates: readonly Candidate[],
  callsByProvider: ReadonlyMap<string, number>,
  excluded: ReadonlySet<string>,
): Candidate[] {
  return candidates
    .filter((c) => !excluded.has(c.probe.provider.id))
    .map((c) => ({ candidate: c, score: scoreOf(c, callsByProvider) }))
    .sort((a, b) => b.score - a.score)
    .map((x) => x.candidate);
}
