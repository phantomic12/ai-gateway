package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"ai-gateway/internal/httperr"
	"ai-gateway/internal/lb"
	"ai-gateway/internal/middleware"
	"ai-gateway/internal/models"

	"github.com/rs/zerolog/log"
)

func qualifiedModelID(providerName, modelID string) string {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return modelID
	}
	if strings.Contains(modelID, "/") {
		return modelID
	}
	providerName = strings.TrimSpace(providerName)
	if providerName == "" {
		return modelID
	}
	return providerName + "/" + modelID
}

func shortModelID(model string) string {
	if i := strings.LastIndex(model, "/"); i >= 0 && i+1 < len(model) {
		return model[i+1:]
	}
	return model
}

func stripProviderPrefix(model string, p *models.Provider) string {
	if p == nil {
		return model
	}
	for _, pre := range []string{p.Name + "/", string(p.Type) + "/"} {
		if pre == "/" {
			continue
		}
		if len(model) > len(pre) && strings.EqualFold(model[:len(pre)], pre) {
			return model[len(pre):]
		}
	}
	return model
}

type bufWriter struct {
	hdr  http.Header
	code int
	buf  bytes.Buffer
}

func newBufWriter() *bufWriter {
	return &bufWriter{hdr: make(http.Header)}
}

func (b *bufWriter) Header() http.Header {
	if b.hdr == nil {
		b.hdr = make(http.Header)
	}
	return b.hdr
}

func (b *bufWriter) WriteHeader(code int) {
	if b.code == 0 {
		b.code = code
	}
}

func (b *bufWriter) Write(p []byte) (int, error) {
	if b.code == 0 {
		b.code = http.StatusOK
	}
	return b.buf.Write(p)
}

func (b *bufWriter) flushTo(w http.ResponseWriter) {
	copyHeader(w.Header(), b.hdr)
	code := b.code
	if code == 0 {
		code = http.StatusOK
	}
	w.WriteHeader(code)
	_, _ = w.Write(b.buf.Bytes())
}

// candidateProviders selects the provider(s) eligible to serve this request.
//
// Routing contract:
//   - Pin always wins: an X-Provider hint, or a qualified model whose first
//     segment names an existing provider/type ("openai/gpt-4o"), routes there
//     exclusively — LB rules are bypassed.
//   - Otherwise a curated lb_rules group (if configured for this model or its
//     alias name) governs: the rule's strategy orders the healthy members
//     (round-robin / random / weighted pick a single serving member;
//     failover returns the whole position-ordered list so proxyCandidates
//     walks it on retriable failures).
//   - Otherwise discovered ownership decides: if exactly one provider
//     advertises the model (qualified or bare slug in provider_models) it
//     serves. If SEVERAL providers advertise the same slug, strict mode
//     returns the whole owner set rotated per request — proxyCandidates
//     spreads load across them and fails over owner-to-owner on retriable
//     failures (a reseller's 403 "access disabled" or 429 shouldn't kill a
//     request other owners can serve). LegacyFallback=true keeps its single
//     health-aware pick for that case.
//   - Otherwise (no owner at all): LegacyFallback=false (default) rejects
//     the request — the caller surfaces model_not_routed;
//     LegacyFallback=true restores the legacy resolution (name/type
//     heuristics, default provider).
//
// Same-provider retries inside proxyWithMetrics remain active per the retry
// policy regardless of strategy.
//
// requestKeyOrg extracts the authenticated gateway key's org scope ("" for
// global keys / dashboard sessions without an org).
func (h *Handler) requestKeyOrg(r *http.Request) string {
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil && k.OrgID != nil {
		return *k.OrgID
	}
	return ""
}

// orgAllows reports whether a key org may use a provider. Policy: global
// (unscoped) providers serve everyone; org-scoped providers serve only that
// org. Previously the proxy path ignored the key's org entirely, so any key
// could route through (and spend) another tenant's provider.
func orgAllows(keyOrg string, p *models.Provider) bool {
	if keyOrg == "" || p == nil || p.OrgID == nil || *p.OrgID == "" {
		return true
	}
	return *p.OrgID == keyOrg
}

func (h *Handler) candidateProviders(rawModel, model, hint, keyOrg string, pred func(*models.Provider) bool) []*models.Provider {
	cands, _ := h.candidateProvidersWithRule(rawModel, model, hint, keyOrg, pred)
	out := make([]*models.Provider, 0, len(cands))
	for _, c := range cands {
		out = append(out, c.Provider)
	}
	return out
}

// candidateProvidersWithRule is candidateProviders plus the curated rule that
// governed selection (nil for pins, legacy resolution, or no-route). Each
// returned candidate carries its own rule member — and with it that member's
// model override — so rules with several options on one provider route
// correctly. Callers pass the rule to proxyCandidates for strategy context
// (e.g. logging); overrides come from the candidates, never by provider id.
func (h *Handler) candidateProvidersWithRule(rawModel, model, hint, keyOrg string, pred func(*models.Provider) bool) ([]lb.Candidate, *lb.Rule) {
	out := make([]lb.Candidate, 0, 1)
	consider := func(p *models.Provider) bool {
		if p == nil || p.ID == "" || (pred != nil && !pred(p)) {
			return false
		}
		out = append(out, lb.Candidate{Provider: p})
		return true
	}

	// 1. Explicit provider hint = hard pin (case-insensitive: display names
	// keep their casing, e.g. "AIHubMix", while callers send any case).
	if hint != "" {
		if p, err := h.ProviderStore.GetByNameCI(hint); err == nil && orgAllows(keyOrg, p) {
			consider(p)
			return out, nil
		}
		if p, err := h.ProviderStore.GetByID(hint); err == nil && orgAllows(keyOrg, p) {
			consider(p)
			return out, nil
		}
	}

	// 2. Qualified-ID pin: "provider/model" where prefix matches a provider
	// name or provider TYPE. Purely namespace-ish models like
	// "openrouter/gpt-4o" don't pin unless such a provider exists.
	if idx := strings.Index(model, "/"); idx > 0 {
		prefix := strings.ToLower(strings.TrimSpace(model[:idx]))
		if prefix != "" {
			if p, err := h.ProviderStore.GetByNameCI(prefix); err == nil && orgAllows(keyOrg, p) && consider(p) {
				return out, nil
			}
			if p, err := h.ProviderStore.GetByType(prefix); err == nil && orgAllows(keyOrg, p) && consider(p) {
				return out, nil
			}
		}
	}

	// 3. Curated LB rule (checked on post-alias model name first, then the
	// raw/alias spelling). The rule's strategy orders healthy members;
	// failover rules hand the whole ordering to proxyCandidates as ordered
	// fallback candidates.
	if h.LB != nil {
		for _, key := range []string{model, rawModel} {
			key = strings.ToLower(strings.TrimSpace(key))
			if key == "" {
				continue
			}
			if rule := h.LB.RuleForModel(key); rule != nil {
				ordered := h.LB.SelectCandidates(rule)
				if len(ordered) == 0 {
					continue
				}
				// Org-scoped keys may only route to providers their org
				// owns (or global ones).
				eligible := make([]lb.Candidate, 0, len(ordered))
				for _, cand := range ordered {
					if orgAllows(keyOrg, cand.Provider) {
						eligible = append(eligible, cand)
					}
				}
				if len(eligible) == 0 {
					continue
				}
				picked := eligible[:1]
				if rule.Strategy == lb.StrategyFailover {
					// Explicit opt-in: walk members in position order on
					// retriable failures.
					picked = eligible
				}
				out = append(out, picked...)
				return out, rule
			}
		}
	}

	// 4. No rule and no pin: serve from discovered ownership. Exactly one
	// provider advertises the slug (qualified or bare) → it serves, in both
	// modes. 2+ providers advertise the SAME slug: strict mode serves the
	// whole owner set, rotated per request — the old "reject" behavior made
	// identical slugs across providers unusable by bare name even though
	// every owner can serve them, and legacy's single health-aware pick had
	// no failover when the picked reseller 403/429'd. The rotated chain
	// spreads load AND walks to the next owner on retriable failures.
	// Qualified IDs (step 2) and X-Provider (step 1) stay the deterministic
	// pins; an LB rule (step 3) always beats implicit ownership.
	owned := h.ownerCandidates(model, keyOrg)
	if len(owned) == 1 {
		consider(owned[0])
		if len(out) > 0 {
			return out, nil
		}
		// pred rejected the sole owner: fall through (strict returns the
		// honest empty result below; legacy heuristics below can't do
		// better — Resolve returns the same provider).
	} else if len(owned) > 1 && !h.LegacyFallback {
		for _, p := range rotateOwners(owned) {
			consider(p)
		}
		if len(out) > 0 {
			return out, nil
		}
	}

	// 5. No rule, no pin, no usable owner: reject in strict mode; in legacy
	// mode a heuristic may still land on the default provider.
	if !h.LegacyFallback {
		return nil, nil
	}
	// Legacy single pick: health-aware round-robin over discovered owners,
	// heuristic ownership by name/type, else default provider. Org-scoped keys
	// resolve through the org-aware resolver (global providers still shared).
	if h.ProviderStore != nil {
		var p *models.Provider
		var err error
		if keyOrg != "" {
			p, err = h.ProviderStore.ResolveWithOrg(model, "", keyOrg)
		} else {
			p, err = h.ProviderStore.Resolve(model, "")
		}
		if err == nil {
			consider(p)
		}
	}
	return out, nil
}

type prepareFn func(p *models.Provider, body []byte) (target, apiKey string, outBody []byte, isAnth bool, err error)

// ownerCandidates lists providers advertising this model (qualified or bare
// slug), org-filtered for scoped keys. ListForModel already orders healthy
// first, then by creation order.
func (h *Handler) ownerCandidates(model, keyOrg string) []*models.Provider {
	if h.ProviderStore == nil || model == "" {
		return nil
	}
	var out []*models.Provider
	for _, p := range h.ProviderStore.ListForModel(model) {
		if orgAllows(keyOrg, p) {
			out = append(out, p)
		}
	}
	return out
}

// ownerRotation spreads ambiguous-owner requests across the owners of a
// shared slug.
var ownerRotation atomic.Uint64

// rotateOwners reorders the owner list so request N starts at owner N%len:
// the first entry serves while healthy, the rest stay as failover targets.
func rotateOwners(owned []*models.Provider) []*models.Provider {
	if len(owned) < 2 {
		return owned
	}
	start := int(ownerRotation.Add(1)-1) % len(owned)
	out := make([]*models.Provider, 0, len(owned))
	out = append(out, owned[start:]...)
	out = append(out, owned[:start]...)
	return out
}

// shouldFailoverFrom reports whether a status justifies trying the next
// candidate provider. The old rule stopped at ANY <500 — including 429 quota
// hiccups and rotated-credential 401/403 — stranding traffic on a dead primary.
func shouldFailoverFrom(status int) bool {
	if status == 0 {
		return true // transport failure
	}
	if status == 429 || status == 401 || status == 403 || status == 408 {
		return true
	}
	return status >= 500
}

func (h *Handler) proxyCandidates(w http.ResponseWriter, r *http.Request, body []byte, isStream bool, model, endpoint, keyPrefix string, start time.Time, candidates []lb.Candidate, rule *lb.Rule, prepare prepareFn) {
	if len(candidates) == 0 {
		httperr.Proxy(w, http.StatusServiceUnavailable, "no provider configured")
		return
	}
	primaryID := candidates[0].Provider.ID

	// One client-facing first-byte budget per client request, shared across
	// the whole candidate chain: every upstream attempt is bounded by the
	// REMAINING budget, so a chain of slow candidates still answers before
	// an edge proxy (Cloudflare ~100s) can 524 the client. Streams that run
	// out of budget get SSE keepalive headers + frames; buffered requests
	// get an honest 504.
	ttfb := newTTFBController(h.Timeouts.TTFB, start)
	defer ttfb.stop()
	w = &keepaliveSafeWriter{ResponseWriter: w, c: ttfb}
	// In-band terminal error once keepalive headers have flowed (a second
	// HTTP status is impossible at that point). /v1/messages clients speak
	// anthropic SSE; everything else OpenAI SSE.
	terminalError := func(status int, msg string) {
		if ttfb.headersCommitted() {
			writeSSEUpstreamError(w, endpoint == "messages", msg)
			return
		}
		httperr.Proxy(w, status, msg)
	}

	var (
		lastBW         *bufWriter
		lastProvider   *models.Provider
		lastFailStatus int
		attempted      bool
		// Diagnostics for the terminal "all attempts failed" response: which
		// provider was tried last and why it failed (bounded upstream error
		// body or transport error text).
		lastProviderID   string
		lastProviderName string
		lastDetail       string
		// chain records every provider that was tried and failed over from,
		// persisted onto the final request_logs row's fallback_chain column.
		chain []providerAttempt
	)

	for idx, cand := range candidates {
		p := cand.Provider
		// Per-member model override: each rule member may send a different
		// model id upstream (e.g. a pinned date version, or a sibling model
		// on the same provider). The override travels on the candidate — a
		// rule may list one provider several times with different models.
		// Pins and legacy routes carry no member, hence no override. The
		// override also feeds usage logging via candModel below.
		candBody, candModel := body, model
		if override := cand.ModelOverrideForRequest(); override != "" && override != model {
			candBody = replaceModelInBody(body, override)
			candModel = override
		}
		target, apiKey, outBody, isAnth, err := prepare(p, candBody)
		if err != nil {
			continue
		}
		attempted = true

		fallbackName := ""
		if p.ID != primaryID && idx > 0 && candidates[0].Provider != nil {
			fallbackName = p.Name
		}
		callOpts := proxyOpts{fallbackFrom: fallbackName, attempts: chain, rule: rule, ttfb: ttfb}

		if isStream {
			outcome := h.proxyWithMetricsOpts(w, r, target, apiKey, outBody, true, candModel, p.ID, keyPrefix, endpoint, start, isAnth, callOpts)
			lastProviderID, lastProviderName = p.ID, p.Name
			lastDetail = outcome.errSnippet
			if outcome.errSnippet == "" {
				lastDetail = outcome.errText
			}
			// Client-caused 5xx (upstream "invalid_request" semantics):
			// relay the upstream's verdict — no failover, no generic 502.
			if outcome.clientCaused {
				terminalError(outboundStatus(outcome.status), truncateDetail(outcome.errSnippet))
				return
			}
			if outcome.committed || !outcome.retriable {
				return // headers already flowed (or hard client-side stop)
			}
			log.Info().Str("candidate", p.Name).Str("model", model).Int("status", outcome.status).Str("detail", truncateDetail(lastDetail)).Msg("stream candidate failed pre-commit, failing over")
			lastFailStatus = outcome.status
			chain = append(chain, providerAttempt{ProviderID: p.ID, Name: p.Name, Status: outcome.status})
			continue
		}

		bw := newBufWriter()
		outcome := h.proxyWithMetricsOpts(bw, r, target, apiKey, outBody, false, candModel, p.ID, keyPrefix, endpoint, start, isAnth, callOpts)
		if !outcome.committed && bw.code == 0 {
			// Terminal pre-commit failure: nothing usable was produced.
			lastProviderID, lastProviderName = p.ID, p.Name
			lastDetail = outcome.errSnippet
			if lastDetail == "" {
				lastDetail = outcome.errText
			}
			// Client-caused 5xx (upstream "invalid_request" semantics):
			// relay the upstream's verdict instead of the retryable-looking
			// generic envelope.
			if outcome.clientCaused {
				terminalError(outboundStatus(outcome.status), truncateDetail(outcome.errSnippet))
				return
			}
			lastFailStatus = outcome.status
			if shouldFailoverFrom(outcome.status) {
				chain = append(chain, providerAttempt{ProviderID: p.ID, Name: p.Name, Status: outcome.status})
				continue
			}
			terminalError(outboundStatus(outcome.status), "upstream unavailable")
			return
		}
		if bw.code > 0 && bw.code < 400 {
			if p.ID != primaryID {
				bw.Header().Set("X-Fallback-Used", p.Name)
				log.Info().Str("fallback_used", p.Name).Str("from", candidates[0].Provider.Name).Str("model", model).Msg("fallback_used")
			}
			bw.flushTo(w)
			return
		}
		lastBW, lastProvider = bw, p
		if shouldFailoverFrom(bw.code) {
			chain = append(chain, providerAttempt{ProviderID: p.ID, Name: p.Name, Status: bw.code})
			continue
		}
		bw.flushTo(w)
		return
	}

	// Every candidate exhausted before committing anything.
	if lastBW != nil {
		if lastProvider != nil && lastProvider.ID != primaryID {
			lastBW.Header().Set("X-Fallback-Used", lastProvider.Name)
		}
		lastBW.flushTo(w)
		return
	}
	// Nothing was even attempted: no eligible candidates, or every candidate
	// failed gateway-side request preparation (e.g. credential decryption).
	// There is no upstream status to relay.
	if !attempted {
		httperr.Proxy(w, http.StatusBadGateway, "no provider available for this model")
		return
	}
	status := http.StatusBadGateway
	if lastFailStatus >= 400 {
		status = outboundStatus(lastFailStatus)
	}
	// The client only ever saw this relayed status with zero context, and no
	// request_logs row was written — failures were invisible in the dashboard.
	// Log the upstream's own answer and persist the failed request.
	log.Error().Str("provider", lastProviderName).Str("model", model).Str("endpoint", endpoint).Int("upstream_status", lastFailStatus).Str("detail", truncateDetail(lastDetail)).Msg("all provider attempts failed")
	h.logRequestExtendedBodies(keyPrefix, lastProviderID, model, endpoint, status, time.Since(start).Milliseconds(), 0, 0, 0, isStream, nil, nil, &logMeta{FallbackChain: marshalAttemptChain(chain)})
	terminalError(status, "all provider attempts failed")
}

// truncateDetail bounds upstream error text for structured logs.
func truncateDetail(s string) string {
	if len(s) > 512 {
		return s[:512] + "…"
	}
	return s
}

// outboundStatus normalizes an upstream status for client-facing responses.
func outboundStatus(s int) int {
	if s >= 400 && s <= 599 {
		return s
	}
	return http.StatusBadGateway
}

// noRouteFor writes the model_not_routed error: a bare model name matched no
// routing rule, no pin matched, and legacy heuristic fallback is disabled.
// The message is actionable — qualified IDs always work without a rule.
func noRouteFor(w http.ResponseWriter, model string) {
	msg := fmt.Sprintf("model %q is not routed: create a routing rule for it, or use provider-qualified IDs like provider/%s (or an X-Provider header)", model, shortModelID(model))
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusNotFound)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"error": map[string]interface{}{
			"message": msg,
			"type":    "invalid_request_error",
			"code":    "model_not_routed",
		},
	})
}

// unroutedModel reports whether an empty candidate list means "no routing
// rule for this model" (legacy fallback disabled) as opposed to the legacy
// resolver simply failing (no providers at all). In the former case callers
// surface model_not_routed; in the latter the pre-existing no-provider 503s.
func (h *Handler) unroutedModel() bool {
	return h.LB != nil && !h.LegacyFallback
}
