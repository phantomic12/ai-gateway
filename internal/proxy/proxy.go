package proxy

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"ai-gateway/internal/apikey"
	"ai-gateway/internal/cache"
	"ai-gateway/internal/catalog"
	"ai-gateway/internal/db"
	"ai-gateway/internal/devin"
	"ai-gateway/internal/httperr"
	"ai-gateway/internal/lb"
	"ai-gateway/internal/middleware"
	"ai-gateway/internal/models"
	"ai-gateway/internal/otel"
	"ai-gateway/internal/provider"
	"ai-gateway/internal/resilience"
	"ai-gateway/internal/translate"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/rs/zerolog/log"
)

// UsageSink receives post-response usage so budgets reconcile against real
// upstream counts instead of only pre-request estimates.
type UsageSink interface {
	RecordUsage(keyPrefix string, orgID *string, tokens int, costUSD float64, ts time.Time)
}

type Handler struct {
	ProviderStore *provider.Store
	CatalogStore  *catalog.Store
	DB            *sql.DB
	Client        *http.Client
	Cache         cache.Cache
	Retry         resilience.RetryPolicy
	Metrics       otel.Metrics
	RateLimiter   middleware.Limiter

	Timeouts TimeoutsConfig
	// LB, when set, routes bare model names through operator-curated
	// strategy groups (see internal/lb). Nil = no curated routing.
	LB *lb.Store
	// LegacyFallback, when true, restores the pre-strategy heuristic model
	// resolution (provider_models ownership round-robin, name heuristics,
	// default provider) for bare model names with no routing rule. Default
	// off: unrouted bare models are rejected with model_not_routed.
	LegacyFallback bool
	// CacheTTLSeconds bounds non-stream completion caching (default 10).
	CacheTTLSeconds int
	// CacheStreams extends the exact-match response cache to streaming
	// requests: a clean stream's client-bound SSE bytes are stored (bounded)
	// and replayed verbatim for identical requests. Opt-in via
	// CACHE_STREAMS=true — all-streaming deployments get zero benefit from
	// the classic non-stream cache.
	CacheStreams bool
	// Usage, when set, records actual per-request token/cost outcomes.
	Usage UsageSink
	// LogBodies enables captured request/response body logging (opt-in),
	// truncated to BodyLogMaxBytes and credential-scrubbed.
	LogBodies       bool
	BodyLogMaxBytes int
	// StreamUsageInject injects stream_options.include_usage for OpenAI-style
	// streams when missing (billing accuracy over maximum compatibility).
	StreamUsageInject bool

	// MaxBodyBytes bounds how much of a client request body the gateway
	// buffers (413 beyond). Zero/negative = DefaultMaxProxyRequestBodyBytes.
	MaxBodyBytes int64

	// keyIDCache maps gateway-key prefix → gateway_keys.id for per-key
	// analytics attribution on request_logs. Warmed on demand; entries are
	// immutable in practice (a prefix maps to one key), so the empty-string
	// miss result (session virtual keys, unknown prefixes) is cached too.
	keyIDMu    sync.RWMutex
	keyIDCache map[string]string
}

func New(ps *provider.Store, db *sql.DB) *Handler {
	tc := DefaultTimeouts()
	return &Handler{
		ProviderStore:   ps,
		DB:              db,
		Client:          GatewayHTTPClient(NewGatewayTransport(tc.UpstreamHeader, true)),
		Timeouts:        tc,
		CacheTTLSeconds: 10,
		BodyLogMaxBytes: 8192,
		MaxBodyBytes:    DefaultMaxProxyRequestBodyBytes,
	}
}

func NewWithCatalog(ps *provider.Store, cs *catalog.Store, db *sql.DB) *Handler {
	h := New(ps, db)
	h.CatalogStore = cs
	return h
}

func (h *Handler) resolveAlias(model string) string {
	// direct alias
	if h.DB != nil {
		var target string
		err := h.DB.QueryRow(db.Q(`SELECT target FROM model_aliases WHERE alias=?`), model).Scan(&target)
		if err == nil && target != "" {
			return target
		}
		// strip opencode/ prefix fallback
		if strings.HasPrefix(model, "opencode/") {
			trimmed := strings.TrimPrefix(model, "opencode/")
			// try alias of trimmed
			err = h.DB.QueryRow(db.Q(`SELECT target FROM model_aliases WHERE alias=?`), trimmed).Scan(&target)
			if err == nil && target != "" {
				return target
			}
			// just return trimmed
			return trimmed
		}
		// also try meta/ prefix for ckff?
	}
	return model
}

func (h *Handler) enforceModelAllowlist(w http.ResponseWriter, r *http.Request, rawModel, resolvedModel string) bool {
	key, ok := middleware.GatewayKeyFromContext(r.Context())
	if !ok || key == nil || len(key.AllowedModels) == 0 {
		return true
	}
	if apikey.IsModelAllowed(key.AllowedModels, resolvedModel) || apikey.IsModelAllowed(key.AllowedModels, rawModel) {
		return true
	}
	httperr.Write(w, http.StatusForbidden, "model not allowed for this key", httperr.TypeInvalid)
	return false
}

// readProxyBody reads a proxied request body capped at the configured proxy
// body limit (maxProxyRequestBodyBytes; see Handler.MaxBodyBytes). Oversized
// bodies are rejected with a clean 413 — previously the ReadAll error was
// discarded and the truncated JSON was relayed upstream, surfacing to the
// client as a misleading upstream 400.
func (h *Handler) readProxyBody(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	limit := h.MaxBodyBytes
	if limit <= 0 {
		limit = DefaultMaxProxyRequestBodyBytes
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, limit))
	if err != nil {
		httperr.Write(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("request body exceeds %d byte limit (MAX_PROXY_BODY_MB)", limit), httperr.TypeInvalid)
		return nil, false
	}
	return body, true
}

func estimateTokens(body []byte) int {
	if len(body) == 0 {
		return 0
	}
	// cheap heuristic: ~4 chars per token; used for TPM pre-check only.
	// Base64 payloads (inline images, audio, files) are excluded BEFORE
	// counting — a 900KB data-URL previously registered ~225K "tokens" and
	// spurious 429'd multimodal clients long before the upstream saw bytes.
	return (len(stripBase64Blobs(body)) + 3) / 4
}

// stripBase64Blobs removes embedded base64 runs (>=256 chars) from a payload.
func stripBase64Blobs(body []byte) []byte {
	const minRun = 256
	isB64 := func(c byte) bool {
		return c == '+' || c == '/' || c == '=' ||
			('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ('0' <= c && c <= '9')
	}
	var out []byte
	i := 0
	for i < len(body) {
		if isB64(body[i]) {
			j := i
			for j < len(body) && isB64(body[j]) {
				j++
			}
			if j-i >= minRun {
				out = append(out, '.')
				i = j
				continue
			}
			out = append(out, body[i:j]...)
			i = j
			continue
		}
		out = append(out, body[i])
		i++
	}
	return out
}

func (h *Handler) checkTPM(w http.ResponseWriter, r *http.Request, body []byte) bool {
	if h.RateLimiter == nil {
		return true
	}
	key, ok := middleware.GatewayKeyFromContext(r.Context())
	if !ok || key == nil || key.RateLimitTPM <= 0 {
		return true
	}
	tokens := estimateTokens(body)
	if tokens == 0 {
		return true
	}
	prefix := key.Prefix
	if prefix == "" {
		prefix = r.Header.Get("X-Gateway-Key-Prefix")
	}
	if prefix == "" {
		return true
	}
	if !h.RateLimiter.AllowTokens(prefix, tokens, key.RateLimitTPM) {
		w.Header().Set("Retry-After", "60")
		httperr.Write(w, http.StatusTooManyRequests, "token rate limit exceeded", httperr.TypeRateLimit)
		return false
	}
	return true
}

// no-op implementations to keep Handler fields wired even when not fully used in proxyWithMetrics
type noopCache struct{}

func (n *noopCache) Get(key string) ([]byte, int, http.Header, bool)                              { return nil, 0, nil, false }
func (n *noopCache) Set(key string, body []byte, status int, headers http.Header, ttlSeconds int) {}
func (n *noopCache) Invalidate(pattern string)                                                    {}

func (h *Handler) cacheOrNoop() cache.Cache {
	if h.Cache != nil {
		return h.Cache
	}
	return &noopCache{}
}

func (h *Handler) retryOrDefault() resilience.RetryPolicy {
	if h.Retry != nil {
		return h.Retry
	}
	return resilience.NewDefaultRetryPolicy()
}

func extractUsage(body []byte) (prompt, completion int) {
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		return 0, 0
	}
	return extractUsageFromMap(m)
}

// extractUsageFromMap is extractUsage over an already-parsed payload, so hot
// paths (SSE harvesting) parse each frame exactly once for billing AND detail.
func extractUsageFromMap(m map[string]interface{}) (prompt, completion int) {
	if m == nil {
		return 0, 0
	}
	if usage, ok := m["usage"].(map[string]interface{}); ok {
		prompt, completion = readUsageTokens(usage)
		// Anthropic prompt caching: cache_read_input_tokens and
		// cache_creation_input_tokens are billed separately from
		// input_tokens. They were previously dropped entirely, systematically
		// under-billing cached-prompt traffic. Count them into the prompt
		// side (priced at the input rate — a close approximation and strictly
		// better than $0).
		if v, ok := usage["cache_read_input_tokens"]; ok {
			prompt += toInt(v)
		}
		if v, ok := usage["cache_creation_input_tokens"]; ok {
			prompt += toInt(v)
		}
		return
	}
	if v, ok := m["usage"]; ok {
		if um, ok := v.(map[string]interface{}); ok {
			if iv, ok := um["input_tokens"]; ok {
				prompt = toInt(iv)
			}
			if cv, ok := um["cache_read_input_tokens"]; ok {
				prompt += toInt(cv)
			}
			if cv, ok := um["cache_creation_input_tokens"]; ok {
				prompt += toInt(cv)
			}
			if ov, ok := um["output_tokens"]; ok {
				completion = toInt(ov)
			}
		}
	}
	return
}

// readUsageTokens reads prompt/completion counts from a usage object across
// dialects WITHOUT any billing folds — cache tokens stay separate here; the
// anthropic cache fold is billing-only and lives in extractUsageFromMap.
func readUsageTokens(usage map[string]interface{}) (prompt, completion int) {
	if v, ok := usage["prompt_tokens"]; ok {
		prompt = toInt(v)
	} else if v, ok := usage["input_tokens"]; ok {
		prompt = toInt(v)
	} else if v, ok := usage["promptTokens"]; ok {
		prompt = toInt(v)
	}
	if v, ok := usage["completion_tokens"]; ok {
		completion = toInt(v)
	} else if v, ok := usage["output_tokens"]; ok {
		completion = toInt(v)
	} else if v, ok := usage["completionTokens"]; ok {
		completion = toInt(v)
	}
	return
}

// usageDetail carries the per-request observability fields that billing
// arithmetic deliberately folds away: the cache-vs-billed prompt split,
// reasoning tokens, and the terminal finish/stop reason.
type usageDetail struct {
	CacheRead    int    `json:"cache_read"`
	CacheWrite   int    `json:"cache_write"`
	Reasoning    int    `json:"reasoning"`
	FinishReason string `json:"finish_reason"`
}

// extractUsageDetail parses the same body shapes extractUsage accepts (chat
// completion / chunk, anthropic message, responses API frames) but returns
// the rich detail WITHOUT the billing folds: cache tokens are reported
// separately, never merged into prompt tokens. Finish reasons are normalized
// to the OpenAI vocabulary (end_turn→stop, max_tokens→length, tool_use→
// tool_calls, refusal→content_filter) so the UI and analytics see one set.
func extractUsageDetail(body []byte) (prompt, completion int, d usageDetail) {
	var m map[string]interface{}
	if err := json.Unmarshal(body, &m); err != nil {
		return 0, 0, d
	}
	return extractUsageDetailFromMap(m)
}

// extractUsageDetailFromMap is extractUsageDetail over an already-parsed payload.
func extractUsageDetailFromMap(m map[string]interface{}) (prompt, completion int, d usageDetail) {
	if m == nil {
		return 0, 0, d
	}
	if usage, ok := m["usage"].(map[string]interface{}); ok {
		prompt, completion = readUsageTokens(usage)
		d = usageDetailFromMap(usage)
	} else if resp, ok := m["response"].(map[string]interface{}); ok {
		// responses API frames wrap usage + status under "response".
		if usage, ok := resp["usage"].(map[string]interface{}); ok {
			prompt, completion = readUsageTokens(usage)
			d = usageDetailFromMap(usage)
		}
		if d.FinishReason == "" {
			d.FinishReason = normalizeFinishReason(resp["status"])
		}
	}
	if d.FinishReason == "" {
		d.FinishReason = finishReasonFromEvent(m)
	}
	return prompt, completion, d
}

// usageDetailFromMap reads the cache/reasoning detail from a usage object.
// OpenAI nests them under *_details (chat: prompt_tokens_details /
// completion_tokens_details; responses API: input_tokens_details /
// output_tokens_details); anthropic uses flat cache_* fields; some
// OpenAI-compatible upstreams put reasoning_tokens flat on usage.
func usageDetailFromMap(usage map[string]interface{}) usageDetail {
	var d usageDetail
	readCached := func(details map[string]interface{}) {
		if v, ok := details["cached_tokens"]; ok && d.CacheRead == 0 {
			d.CacheRead = toInt(v)
		}
	}
	if pd, ok := usage["prompt_tokens_details"].(map[string]interface{}); ok {
		readCached(pd)
	}
	if pd, ok := usage["input_tokens_details"].(map[string]interface{}); ok {
		readCached(pd)
	}
	if cd, ok := usage["completion_tokens_details"].(map[string]interface{}); ok {
		if v, ok := cd["reasoning_tokens"]; ok {
			d.Reasoning = toInt(v)
		}
	}
	if cd, ok := usage["output_tokens_details"].(map[string]interface{}); ok {
		if v, ok := cd["reasoning_tokens"]; ok && d.Reasoning == 0 {
			d.Reasoning = toInt(v)
		}
	}
	if v, ok := usage["cache_read_input_tokens"]; ok {
		d.CacheRead = toInt(v)
	}
	if v, ok := usage["cache_creation_input_tokens"]; ok {
		d.CacheWrite = toInt(v)
	}
	if v, ok := usage["reasoning_tokens"]; ok && d.Reasoning == 0 {
		d.Reasoning = toInt(v)
	}
	return d
}

// finishReasonFromEvent extracts and normalizes the terminal reason from a
// full event payload: anthropic messages carry stop_reason at top level,
// chat completions under choices[0].finish_reason, responses API under
// response.status. First non-empty wins.
func finishReasonFromEvent(m map[string]interface{}) string {
	if m == nil {
		return ""
	}
	if sr, ok := m["stop_reason"]; ok {
		if s := normalizeFinishReason(sr); s != "" {
			return s
		}
	}
	if choices, ok := m["choices"].([]interface{}); ok && len(choices) > 0 {
		if c0, ok := choices[0].(map[string]interface{}); ok {
			return normalizeFinishReason(c0["finish_reason"])
		}
	}
	if st, ok := m["status"]; ok {
		return normalizeFinishReason(st)
	}
	return ""
}

// normalizeFinishReason maps provider-native terminal reasons onto the OpenAI
// vocabulary; unknown/empty values pass through unchanged (or empty).
func normalizeFinishReason(v interface{}) string {
	s, _ := v.(string)
	switch s {
	case "":
		return ""
	case "completed":
		// responses API happy-path lifecycle status
		return "stop"
	case "in_progress", "queued":
		// non-terminal lifecycle statuses are not terminal reasons
		return ""
	case "end_turn", "stop", "stop_sequence":
		return "stop"
	case "max_tokens", "length", "max_output_tokens":
		return "length"
	case "tool_use", "tool_calls", "function_call":
		return "tool_calls"
	case "refusal", "content_filter", "safety":
		return "content_filter"
	default:
		return s
	}
}

func toInt(v interface{}) int {
	switch x := v.(type) {
	case float64:
		if x < 0 {
			return 0
		}
		return int(x)
	case int:
		if x < 0 {
			return 0
		}
		return x
	case int64:
		if x < 0 {
			return 0
		}
		return int(x)
	case json.Number:
		i, _ := x.Int64()
		if i < 0 {
			return 0
		}
		return int(i)
	}
	return 0
}

// DefaultMaxProxyRequestBodyBytes bounds how much of a client request body
// the gateway will buffer by default. Upstream LLM payloads can legitimately
// reach tens of MB (large multimodal contexts, agentic sessions carrying
// whole codebases in context), but an unbounded read let a single
// authenticated request exhaust gateway memory (DoS for every tenant).
// Operator-configurable via MAX_PROXY_BODY_MB (0 disables the cap).
const DefaultMaxProxyRequestBodyBytes = 64 << 20 // 64 MiB

func cacheKeyFor(endpoint, model string, body []byte, scope string) string {
	sum := sha256.Sum256(append([]byte(endpoint+"\n"+scope+"\n"+model+"\n"), body...))
	return "post:" + hex.EncodeToString(sum[:])
}

// cacheScope identifies the tenant (and resolved provider) a cached completion
// belongs to. Without it any key whose allowlist permitted the same model could
// be served another tenant's byte-identical cached completion.
func cacheScopeFor(r *http.Request, providerID string) string {
	prefix := "anon"
	org := ""
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil {
		if k.Prefix != "" {
			prefix = k.Prefix
		}
		if k.OrgID != nil {
			org = *k.OrgID
		}
	}
	return prefix + "|" + org + "|" + providerID
}

func modelsCacheKey(r *http.Request) string {
	prefix := "anon"
	allow := ""
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil {
		if k.Prefix != "" {
			prefix = k.Prefix
		}
		if len(k.AllowedModels) > 0 {
			allow = strings.Join(k.AllowedModels, ",")
		}
	}
	return "models:" + prefix + ":" + allow
}

func retryAfterDelay(hdr http.Header, fallback time.Duration) time.Duration {
	if hdr == nil {
		return fallback
	}
	ra := hdr.Get("Retry-After")
	if ra == "" {
		return fallback
	}
	secs, err := strconv.Atoi(strings.TrimSpace(ra))
	if err != nil || secs < 0 {
		return fallback
	}
	d := time.Duration(secs) * time.Second
	if d > 5*time.Second {
		d = 5 * time.Second
	}
	return d
}

// hopByHopHeaders are response headers owned by the upstream CONNECTION and
// must never be forwarded to gateway clients (RFC 7230 §6.1).
var hopByHopHeaders = map[string]bool{
	"connection": true, "proxy-connection": true, "keep-alive": true,
	"proxy-authenticate": true, "proxy-authorization": true, "te": true,
	"trailer": true, "transfer-encoding": true, "upgrade": true,
}

func copyHeader(dst, src http.Header) {
	// Headers named in a Connection: header are hop-by-hop too. Build a
	// per-call set: the previous version WROTE these tokens into the shared
	// hopByHopHeaders map, which is a data race under concurrent responses
	// (Go fatal: concurrent map writes) and permanently blacklisted headers
	// for the process lifetime.
	connTokens := map[string]bool{}
	for _, tok := range strings.Split(src.Get("Connection"), ",") {
		if t := strings.ToLower(strings.TrimSpace(tok)); t != "" {
			connTokens[t] = true
		}
	}
	for k, vv := range src {
		if hopByHopHeaders[strings.ToLower(k)] || connTokens[strings.ToLower(k)] {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func (h *Handler) serveCacheHit(w http.ResponseWriter, body []byte, status int, headers http.Header) {
	copyHeader(w.Header(), headers)
	w.Header().Set("X-Cache", "HIT")
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	if h.Metrics != nil {
		h.Metrics.IncCacheHit(true)
	}
	if status == 0 {
		status = http.StatusOK
	}
	w.WriteHeader(status)
	w.Write(body)
}

func (h *Handler) writeJSONCached(w http.ResponseWriter, r *http.Request, cacheKey string, ttl int, payload any) {
	b, _ := json.Marshal(payload)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	w.Write(b)
	if cacheKey != "" && len(b) <= 1<<20 {
		h.cacheOrNoop().Set(cacheKey, b, http.StatusOK, http.Header{"Content-Type": []string{"application/json"}}, ttl)
	}
	// Model-list requests carry X-Cache like every other gateway response;
	// log them so the cache disposition shows in reporting too (free
	// metadata call: zero tokens, zero cost).
	h.logRequestExtendedBodies(r.Header.Get("X-Gateway-Key-Prefix"), "", "", "models", http.StatusOK, 0, 0, 0, 0, false, nil, nil, &logMeta{CacheStatus: "miss"})
}

// upstreamForwardHeaders is an ALLOWLIST of client headers proxied to
// upstreams. The old code used a deny-list for auth/gateway hops only, so
// cookies, arbitrary SDK telemetry, anthropic-version (against OpenAI targets)
// and other residue rode along when fallback switched providers mid-request.
// x-opencode-session is required by OpenCode Go/Zen (see
// https://opencode.ai/docs/go/#where-can-i-use-it): without it the upstream
// rejects requests with 400 MissingSessionID on /responses (and 500s on the
// translated /chat/completions fallback). The gateway must not strip it.
var upstreamForwardHeaders = map[string]bool{
	"accept":             true,
	"accept-language":    true,
	"user-agent":         true,
	"x-request-id":       true,
	"x-opencode-session": true,
}

// opencodeSessionForRequest is the stable session ID the gateway sends to
// OpenCode Go/Zen upstreams when the client did not supply one. The upstream
// hard-requires x-opencode-session (400 MissingSessionID otherwise) and asks
// for a stable value per conversation so it can optimize routing and prompt
// caching: a per-request UUID would defeat that, so this reuses the
// client's x-request-id, falling back to a fixed gateway identifier.
func opencodeSessionForRequest(r *http.Request) string {
	if v := r.Header.Get("x-opencode-session"); v != "" {
		return v
	}
	if v := r.Header.Get("x-request-id"); v != "" {
		return v
	}
	return "ai-gateway"
}

// isOpencodeTarget reports whether the upstream URL is an OpenCode Go/Zen
// endpoint that hard-requires the x-opencode-session header.
func isOpencodeTarget(targetURL string) bool {
	l := strings.ToLower(targetURL)
	return strings.Contains(l, "opencode.ai/zen")
}

// errSnippet condenses an upstream error body to a single short log-safe
// string: control characters stripped, capped to 200 bytes.
func errSnippet(b []byte) string {
	s := strings.Map(func(r rune) rune {
		if r < 0x20 && r != '\t' {
			return -1
		}
		return r
	}, strings.TrimSpace(string(b)))
	if len(s) > 200 {
		s = s[:200]
	}
	return s
}

// deleteResponsesKey removes a top-level key from a Responses API body,
// returning nil when the key is absent (no re-marshal needed).
func deleteResponsesKey(body []byte, key string) []byte {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil || m == nil {
		return nil
	}
	if _, ok := m[key]; !ok {
		return nil
	}
	delete(m, key)
	out, err := json.Marshal(m)
	if err != nil {
		return nil
	}
	return out
}

// azureAPIVersion is the default Azure OpenAI API version used for
// deployment-style URLs.
const azureAPIVersion = "2024-06-01"

// upstreamTarget builds the upstream URL for an OpenAI-style action
// ("/chat/completions", "/completions", "/embeddings", "/models").
// Azure providers need deployment-shaped paths plus an api-version query;
// the request builder detects that shape and switches the auth header from
// "Authorization: Bearer" to "api-key" accordingly.
func upstreamTarget(baseURL, action, model string, isAzure bool) string {
	base := strings.TrimRight(baseURL, "/")
	if !isAzure {
		return base + action
	}
	if strings.Contains(base, "/openai/deployments/") {
		return base + action + "?api-version=" + azureAPIVersion
	}
	return base + "/openai/deployments/" + url.PathEscape(model) + action + "?api-version=" + azureAPIVersion
}

func (h *Handler) newUpstreamRequest(ctx context.Context, r *http.Request, targetURL, apiKey string, body []byte, isStream, isAnthropicUpstream bool) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, r.Method, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	for k, vv := range r.Header {
		lk := strings.ToLower(k)
		switch lk {
		case "authorization", "host", "content-length", "cookie":
			continue
		}
		// Anthropic feature headers must survive the hop to Anthropic
		// upstreams: every beta-gated capability (mcp-client,
		// fine-grained-tool-streaming, token-efficient-tools, output-128k, …)
		// silently disappears if anthropic-beta is dropped, and clients lose
		// the ability to pin an anthropic-version. Never forwarded to
		// OpenAI-style upstreams, where they are unknown residue.
		forward := upstreamForwardHeaders[lk] ||
			(isAnthropicUpstream && (lk == "anthropic-beta" || lk == "anthropic-version"))
		if !forward {
			continue
		}
		for _, v := range vv {
			req.Header.Add(k, v)
		}
	}
	if req.Header.Get("x-opencode-session") == "" && isOpencodeTarget(targetURL) {
		// OpenCode Go/Zen hard-requires a session ID (400 MissingSessionID
		// otherwise) and asks for a stable value per conversation; inject the
		// gateway default when the client sent none (see opencodeSessionForRequest).
		req.Header.Set("x-opencode-session", opencodeSessionForRequest(r))
	}
	if isAnthropicUpstream {
		req.Header.Set("x-api-key", apiKey)
		// Honor a client-pinned version; default only when absent.
		if req.Header.Get("anthropic-version") == "" {
			req.Header.Set("anthropic-version", "2023-06-01")
		}
	} else if strings.Contains(targetURL, "/openai/deployments/") {
		// Azure: api-key header, no Bearer.
		req.Header.Set("api-key", apiKey)
	} else {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	req.Header.Set("Content-Type", "application/json")
	// Never relay a Go-derived User-Agent to an upstream. Go's net/http
	// default ("Go-http-client/1.1") is blocked with HTTP 403 by the
	// Cloudflare WAF fronting some providers (inference-api.nousresearch.com
	// — error 1010 "you have been blocked"). Clients that send no UA, or
	// send the Go default verbatim, poison the upstream call; substitute the
	// gateway's identity (see models.GatewayUserAgent). A real client UA
	// (curl, httpx, browser) is left intact.
	if ua := req.Header.Get("User-Agent"); ua == "" || strings.HasPrefix(ua, "Go-http-client") {
		req.Header.Set("User-Agent", models.GatewayUserAgent)
	}
	if isStream {
		req.Header.Set("Accept", "text/event-stream")
	} else {
		req.Header.Set("Accept", "application/json")
	}
	return req, nil
}

// sleepCtx sleeps for d but wakes immediately on ctx cancellation.
func sleepCtx(ctx context.Context, d time.Duration) {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
	case <-ctx.Done():
	}
}

// proxyOpts carries per-invocation metadata through candidate fallback chains.
type proxyOpts struct {
	// fallbackFrom, when non-empty, names the primary provider so the
	// X-Fallback-Used response header can be applied BEFORE any bytes are
	// committed (streams included).
	fallbackFrom string
	// translatedResponses marks that the outbound body was a translation of an
	// OpenAI Responses request onto a chat/anthropic backend, so the returned
	// JSON must be converted back into Responses shape (non-streaming).
	translatedResponses bool
	// attempts records every provider tried before the one that ultimately
	// served the request, for the request_logs fallback_chain column.
	attempts []providerAttempt
	// rule, when non-nil, is the curated lb rule that produced the candidate
	// list — enabling per-member model overrides. Nil for pinned/legacy
	// routes, where overrides never apply.
	rule *lb.Rule
	// ttfb, when non-nil, is the client-facing first-byte controller owned by
	// proxyCandidates for the WHOLE candidate chain (one budget per client
	// request, one potential keepalive commit). Nil = direct call; the handler
	// creates a request-local controller instead.
	ttfb *ttfbController
	// prevResponseID links a translated /v1/responses turn to its stored
	// predecessor so the completed turn can extend the chain. Empty for
	// first turns and non-responses endpoints.
	prevResponseID string
}

// providerAttempt is one tried-and-failed-over provider in a fallback chain.
type providerAttempt struct {
	ProviderID string `json:"provider_id"`
	Name       string `json:"name"`
	Status     int    `json:"status"`
}

// marshalAttemptChain renders the attempted chain as JSON for request_logs;
// empty when nothing was attempted (no chain to report).
func marshalAttemptChain(attempts []providerAttempt) string {
	if len(attempts) == 0 {
		return ""
	}
	out, err := json.Marshal(attempts)
	if err != nil {
		return ""
	}
	return string(out)
}

// attemptOutcome describes how one proxyWithMetrics invocation ended.
type attemptOutcome struct {
	// committed: response bytes were written to the client (headers sent).
	committed bool
	// status: best-known upstream HTTP status (0 = transport-level failure).
	status int
	// retriable: caller may advance to another candidate/provider.
	retriable bool
	// errSnippet: bounded upstream error body (5xx/429) or transport error
	// text, for operator diagnostics on terminal failure. Empty on success.
	errSnippet string
	// clientCaused: the upstream 5xx actually carried client-error semantics
	// (e.g. new-api "invalid_request" for JSON type mismatches). The terminal
	// path relays the upstream body + status instead of the generic envelope.
	clientCaused bool
	// errText: transport-level error string (Client.Do failure). Empty when
	// the upstream answered with a status.
	errText string
}

// proxyWithMetrics handles both anthropic and openai upstreams correctly.
//
// Retry/fallback rules (post-hardening):
//   - NOTHING reaches the client until a usable upstream response exists.
//     Streams retry/failover exactly like buffered calls as long as headers
//     have not been committed — the gateway buffers only headers (never body
//     bytes) while deciding.
//   - Mid-stream failures terminate the SSE channel with protocol-correct
//     error frames ([DONE]/anthropic error event) and record honest outcomes.
//
// isStream4xxRelayStatus reports whether an upstream status on a streaming
// request should be relayed to the client verbatim (status + body) instead of
// entering the SSE pump: plain client errors are the caller's fault and must
// reach it unmangled. 429 and 5xx stay in the retry/failure machinery.
func isStream4xxRelayStatus(status int) bool {
	return status >= 400 && status < 500
}

// isClientCaused5xx detects 5xx responses that actually carry client-error// semantics. new-api style upstreams (observed live on ckff.dev) answer
// deterministic JSON type mismatches (n: 2.5, temperature: "hot", stop with
// non-string elements) with HTTP 500 + `"code":"invalid_request"`, raw Go
// "cannot unmarshal" decode errors, or outright "Panic detected" crashes.
// All three are deterministic for the identical request body: retrying
// cannot succeed.
func isClientCaused5xx(body []byte) bool {
	var probe struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &probe) != nil || len(probe.Error) == 0 {
		return false
	}
	// error may be a string or an object with code/type/message fields.
	var eo struct {
		Code    string `json:"code"`
		Type    string `json:"type"`
		Message string `json:"message"`
	}
	if json.Unmarshal(probe.Error, &eo) == nil {
		code := strings.ToLower(eo.Code)
		typ := strings.ToLower(eo.Type)
		msg := strings.ToLower(eo.Message)
		if code == "invalid_request" || code == "invalid_request_error" ||
			code == "convert_request_failed" || code == "model_not_found" ||
			typ == "invalid_request_error" {
			return true
		}
		return strings.Contains(msg, "cannot unmarshal") ||
			strings.Contains(msg, "panic detected") ||
			strings.Contains(msg, "failed to decode base64") ||
			strings.Contains(msg, "illegal base64 data")
	}
	var es string
	if json.Unmarshal(probe.Error, &es) == nil {
		ls := strings.ToLower(es)
		return strings.Contains(ls, "invalid_request") ||
			strings.Contains(ls, "convert_request_failed") ||
			strings.Contains(ls, "cannot unmarshal") ||
			strings.Contains(ls, "panic detected") ||
			strings.Contains(ls, "failed to decode base64") ||
			strings.Contains(ls, "illegal base64 data")
	}
	return false
}

func (h *Handler) proxyWithMetrics(w http.ResponseWriter, r *http.Request, targetURL string, apiKey string, body []byte, isStream bool, model string, providerID string, keyPrefix string, endpoint string, start time.Time, isAnthropicUpstream bool) attemptOutcome {
	return h.proxyWithMetricsOpts(w, r, targetURL, apiKey, body, isStream, model, providerID, keyPrefix, endpoint, start, isAnthropicUpstream, proxyOpts{})
}

func (h *Handler) proxyWithMetricsOpts(w http.ResponseWriter, r *http.Request, targetURL, apiKey string, body []byte, isStream bool, model, providerID, keyPrefix, endpoint string, start time.Time, isAnthropicUpstream bool, opts proxyOpts) attemptOutcome {
	c := h.cacheOrNoop()
	retry := h.retryOrDefault()

	cacheTTL := h.CacheTTLSeconds
	if cacheTTL <= 0 {
		cacheTTL = 10
	}

	// cacheMaxBodyBytes bounds the REQUEST body size eligible for the exact-
	// match cache. Agentic prompts run large; a 1 MiB cap excluded a real
	// share of traffic. Cached VALUES remain response-bounded (1 MiB), so
	// this only widens which requests may consult/participate.
	const cacheMaxBodyBytes = 4 << 20

	// cacheStatus records this request's disposition against the response
	// cache on every log row: 'hit', 'miss' (eligible, consulted, absent) or
	// 'bypass' (not eligible: oversize body, streaming with the stream cache
	// off). The X-Cache response header mirrors it.
	cacheStatus := "bypass"
	if len(body) <= cacheMaxBodyBytes {
		if !isStream || h.CacheStreams {
			cacheStatus = "miss"
		}
	}

	cacheKey := ""
	if !isStream && len(body) <= cacheMaxBodyBytes {
		cacheKey = cacheKeyFor(endpoint, model, body, cacheScopeFor(r, providerID))
		if cached, status, headers, ok := c.Get(cacheKey); ok && status >= 200 && status < 400 {
			h.serveCacheHit(w, cached, status, headers)
			// Stamp the row as a gateway cache hit so reporting can separate
			// cached responses from fresh upstream calls (token counts stay
			// zero: cached responses are neither re-billed nor re-metered).
			h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, status, time.Since(start).Milliseconds(), 0, 0, 0, false, nil, nil, &logMeta{CacheHit: true, CacheStatus: "hit"})
			return attemptOutcome{committed: true, status: status}
		}
		if h.Metrics != nil {
			h.Metrics.IncCacheHit(false)
		}
	}

	// Streaming response cache (opt-in, CACHE_STREAMS=true): replay the exact
	// SSE bytes a previous identical request received. The key is salted with
	// the stream marker so a cached stream can never be served to a
	// non-streaming client and vice versa.
	var streamCache *streamCacheCollector
	if isStream && h.CacheStreams && len(body) <= cacheMaxBodyBytes {
		sKey := cacheKeyFor(endpoint+"|stream", model, body, cacheScopeFor(r, providerID))
		if cached, status, headers, ok := c.Get(sKey); ok && status >= 200 && status < 400 {
			h.serveCacheHit(w, cached, status, headers)
			if f, canFlush := w.(http.Flusher); canFlush {
				f.Flush()
			}
			h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, status, time.Since(start).Milliseconds(), 0, 0, 0, true, nil, nil, &logMeta{CacheHit: true, CacheStatus: "hit"})
			return attemptOutcome{committed: true, status: status}
		}
		if h.Metrics != nil {
			h.Metrics.IncCacheHit(false)
		}
		streamCache = &streamCacheCollector{key: sKey, ttl: cacheTTL, max: 1 << 20}
	}

	ctx := r.Context()
	var cancelFn context.CancelFunc
	if h.Timeouts.RequestTotal > 0 {
		ctx, cancelFn = context.WithTimeout(ctx, h.Timeouts.RequestTotal)
		defer cancelFn()
	}

	// Client-facing first-byte budget (default 85s, under Cloudflare's ~100s
	// 524 window). The gateway is silent pre-commit by design; without this
	// budget a slow upstream burns the whole window, the edge 524s, and the
	// gateway later logs a 499 for a request that was still healthy.
	// proxyCandidates owns one controller per client request and passes it in
	// opts; direct callers create a request-local one here. In both cases the
	// writer is wrapped so heartbeat frames and stream writes share one lock.
	ttfb := opts.ttfb
	if ttfb == nil {
		ttfb = newTTFBController(h.Timeouts.TTFB, start)
		defer ttfb.stop()
		w = &keepaliveSafeWriter{ResponseWriter: w, c: ttfb}
	}
	defer ttfb.stop()

	applyFallbackHeader := func(hdr http.Header) {
		if opts.fallbackFrom != "" {
			hdr.Set("X-Fallback-Used", opts.fallbackFrom)
		}
	}

	var (
		lastStatus int
		lastErr    error
		lastHdr    http.Header
		lastBody   []byte
		// lastErrSnippet holds a bounded sample of the most recent upstream
		// error body so terminal failures can say WHY the provider 5xx'd.
		lastErrSnippet []byte
	)

	for attempt := 0; ; attempt++ {
		// Budget already spent by earlier candidates/retries and this call is
		// buffered (nothing in-band to keep alive): fail fast instead of
		// piling another silent attempt onto a doomed request.
		if !isStream && ttfb.expired() {
			httperr.Proxy(w, http.StatusGatewayTimeout, "upstream is not responding (first-byte timeout)")
			h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, http.StatusGatewayTimeout, time.Since(start).Milliseconds(), 0, 0, 0, false, nil, nil, &logMeta{CacheStatus: cacheStatus})
			return attemptOutcome{committed: true, status: http.StatusGatewayTimeout}
		}
		req, err := h.newUpstreamRequest(ctx, r, targetURL, apiKey, body, isStream, isAnthropicUpstream)
		if err != nil {
			if ttfb.headersCommitted() {
				writeSSEUpstreamError(w, sseClientDialect(endpoint, isAnthropicUpstream), "failed to create upstream request")
				return attemptOutcome{committed: true, status: http.StatusInternalServerError}
			}
			httperr.Write(w, http.StatusInternalServerError, "failed to create upstream request", httperr.TypeProxy)
			return attemptOutcome{committed: true, status: http.StatusInternalServerError}
		}
		// Bound THIS attempt's wait for upstream response headers by the
		// client's remaining first-byte budget (not the body — a successful
		// header arrival disarms the watchdog, leaving long streams free).
		attemptCtx, attemptCancel := context.WithCancel(ctx)
		defer attemptCancel()
		req = req.WithContext(attemptCtx)
		ttfbTimer := armTTFBWatchdog(ttfb, attemptCancel)
		resp, err := h.Client.Do(req)
		if ttfbTimer != nil {
			ttfbTimer.Stop()
		}
		if err != nil {
			// First-byte budget spent while the upstream stayed silent: turn
			// the transport timeout into a controlled outcome before the edge
			// (Cloudflare 524) does it for us.
			if ttfb.expired() {
				lastErr, lastStatus = err, 0
				if ttfb.headersCommitted() {
					// Keepalive umbra already flowing (earlier attempt hit
					// the budget; this one died too). A real status line is
					// impossible — retry under the umbra or end in-band.
					if retry.ShouldRetry(attempt, retryableCode(0)) {
						sleepCtx(ctx, retryAfterDelay(nil, retry.Backoff(attempt)))
						continue
					}
					writeSSEUpstreamError(w, sseClientDialect(endpoint, isAnthropicUpstream), "upstream is not responding; request aborted before first token")
					return attemptOutcome{committed: true, status: 0, retriable: false, errText: "ttfb budget exhausted (stream keepalive committed)"}
				}
				if isStream {
					// Streams: commit SSE headers + keepalive so the client
					// (and any edge proxy) see bytes, then restart a fresh
					// upstream attempt under the keepalive umbra.
					ttfb.commitKeepalive(w, opts.fallbackFrom)
					if retry.ShouldRetry(attempt, retryableCode(0)) {
						sleepCtx(ctx, retryAfterDelay(nil, retry.Backoff(attempt)))
						continue
					}
					writeSSEUpstreamError(w, sseClientDialect(endpoint, isAnthropicUpstream), "upstream is not responding; request aborted before first token")
					return attemptOutcome{committed: true, status: 0, retriable: false, errText: "ttfb budget exhausted (stream keepalive committed)"}
				}
				// Buffered (or exhausted retries): an honest 504 that edges
				// pass through beats a synthesized 524 the gateway never sees.
				httperr.Proxy(w, http.StatusGatewayTimeout, "upstream is not responding (first-byte timeout)")
				h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, http.StatusGatewayTimeout, time.Since(start).Milliseconds(), 0, 0, 0, isStream, nil, nil, &logMeta{CacheStatus: cacheStatus})
				return attemptOutcome{committed: true, status: http.StatusGatewayTimeout}
			}
		}
		if err != nil || (resp != nil && resp.StatusCode >= 500) || (resp != nil && resp.StatusCode == 429) {
			// Upstream unusable and NOTHING has been committed yet — both
			// streams and buffered requests can recover here.
			status := 0
			var retryHdr http.Header
			var retryBody []byte
			clientCaused := false
			if err == nil {
				status = resp.StatusCode
				// Retain headers + body: the terminal response after the last
				// attempt must relay the upstream's real error (quota message,
				// Retry-After) instead of a generic gateway envelope.
				retryHdr = resp.Header.Clone()
				retryBody, _ = io.ReadAll(io.LimitReader(resp.Body, 1<<20))
				resp.Body.Close()
				// Some upstreams (new-api style, seen live on ckff.dev) answer
				// JSON type mismatches — n: 2.5, temperature: "hot" — with a
				// 500 carrying code "invalid_request". That is deterministic
				// CLIENT error semantics: retrying cannot succeed. Classify,
				// then skip retry and fail the request directly.
				clientCaused = isClientCaused5xx(retryBody)
			}
			lastErr, lastStatus, lastHdr, lastBody = err, status, retryHdr, retryBody
			// Keep a bounded sample of the upstream's error output so the
			// terminal "attempts failed" path can log WHY the provider 5xx'd
			// (quota text, model errors, HTML error pages — truncated).
			if len(retryBody) > 0 {
				lastErrSnippet = retryBody
				if len(lastErrSnippet) > 2048 {
					lastErrSnippet = lastErrSnippet[:2048]
				}
			}
			if clientCaused {
				// Relay the upstream's invalid_request verdict as-is: the
				// client needs the message, not a retryable-looking 500
				// after pointless backoff.
				return attemptOutcome{committed: false, status: status, retriable: false, errSnippet: string(lastErrSnippet), clientCaused: true}
			}
			if retry.ShouldRetry(attempt, retryableCode(status)) {
				sleepCtx(ctx, retryAfterDelay(retryHdr, retry.Backoff(attempt)))
				continue
			}
			if err != nil {
				// Distinguish client-cancelled from genuinely dead upstream.
				if ctx.Err() != nil {
					return attemptOutcome{committed: false, status: status, retriable: false, errText: err.Error()}
				}
				return attemptOutcome{committed: false, status: 0, retriable: true, errText: err.Error()}
			}
			return attemptOutcome{committed: false, status: status, retriable: true, errSnippet: string(lastErrSnippet)}
		}

		lastStatus = resp.StatusCode
		lastHdr = resp.Header.Clone()
		lastErr = nil

		// Upstream 4xx (client error, non-429) on a streaming request: relay
		// the upstream's status + body verbatim. Previously this fell into
		// pumpStream, whose non-SSE first-chunk guard laundered the upstream's
		// informative 400 ("max_tokens: field required", bad model, …) into a
		// generic 502. A 4xx is the caller's error, not the provider's.
		if isStream && isStream4xxRelayStatus(resp.StatusCode) {
			errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
			resp.Body.Close()
			if ttfb.headersCommitted() {
				// Keepalive headers already flowed: a real status line is
				// impossible. Deliver the verdict in-band as an SSE error.
				ttfb.stop()
				writeSSEUpstreamError(w, sseClientDialect(endpoint, isAnthropicUpstream),
					"upstream error "+strconv.Itoa(resp.StatusCode)+": "+upstreamErrorMessage(errBody, "client error"))
				h.logRequestExtended(keyPrefix, providerID, model, endpoint, resp.StatusCode, time.Since(start).Milliseconds(), 0, 0, 0, true)
				return attemptOutcome{committed: true, status: resp.StatusCode}
			}
			copyHeader(w.Header(), resp.Header)
			applyFallbackHeader(w.Header())
			w.WriteHeader(resp.StatusCode)
			_, _ = w.Write(errBody)
			h.logRequestExtended(keyPrefix, providerID, model, endpoint, resp.StatusCode, time.Since(start).Milliseconds(), 0, 0, 0, true)
			return attemptOutcome{committed: true, status: resp.StatusCode}
		}

		if isStream {
			h.pumpStream(w, r, req.Context(), resp, model, keyPrefix, providerID, endpoint, start, isAnthropicUpstream, applyFallbackHeader, ttfb, streamCache, cacheStatus)
			ttfb.stop()
			return attemptOutcome{committed: true, status: lastStatus}
		}

		bodyBytes, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		lastBody = bodyBytes
		if readErr != nil && len(bodyBytes) == 0 {
			lastErr, lastStatus = readErr, 0
			if retry.ShouldRetry(attempt, 0) {
				sleepCtx(ctx, retryAfterDelay(nil, retry.Backoff(attempt)))
				continue
			}
			return attemptOutcome{committed: false, status: 0, retriable: true}
		}
		// Strict upstreams (new-api style) can answer 200 with an EMPTY body
		// when their channel dies mid-request. Relaying that hands the client
		// a "success" with no content — and caches it. Treat it as a
		// transport-level failure: retry, then fail over / 502.
		if lastStatus == 200 && len(bodyBytes) == 0 {
			lastErr, lastStatus = errEmptyUpstreamBody, 0
			lastBody = nil
			if retry.ShouldRetry(attempt, 0) {
				sleepCtx(ctx, retryAfterDelay(nil, retry.Backoff(attempt)))
				continue
			}
			return attemptOutcome{committed: false, status: 0, retriable: true, errText: errEmptyUpstreamBody.Error()}
		}
		if retry.ShouldRetry(attempt, resp.StatusCode) {
			sleepCtx(r.Context(), retryAfterDelay(resp.Header, retry.Backoff(attempt)))
			continue
		}
		// Fragmented tool_calls repair (buffered responses only): some
		// upstreams split one call's arguments across entries with empty
		// ids, and standard harnesses parse each entry and die on the
		// fragments. Streaming deltas are legitimately fragmented and
		// stay verbatim for the client to accumulate.
		if !isStream && lastStatus == 200 && len(bodyBytes) > 0 {
			if fixed, did := mergeFragmentedToolCalls(bodyBytes); did {
				bodyBytes = fixed
				lastBody = fixed
			}
		}
		break
	}

	// Terminal failure path: every attempt failed pre-commit.
	if lastStatus == 0 && len(lastBody) == 0 {
		log.Error().Err(lastErr).Str("target", targetURL).Msg("upstream error")
		httperr.Proxy(w, http.StatusBadGateway, "upstream unavailable")
		h.logRequestExtended(keyPrefix, providerID, model, endpoint, 502, time.Since(start).Milliseconds(), 0, 0, 0, isStream)
		return attemptOutcome{committed: true, status: http.StatusBadGateway}
	}

	{
		pt, ct := extractUsage(lastBody)
		if pt == 0 && ct == 0 {
			pt, ct = extractAnthropicUsageFromSSEChunk(lastBody)
		}
		cost := h.costForModel(model, pt, ct)
		outBody := lastBody

		needsChatShape := endpoint == "chat.completions" && isAnthropicUpstream && lastStatus == 200
		if needsChatShape {
			translated := anthropicToOpenAIChatResponse(lastBody, model)
			if translated != nil {
				pt2, ct2 := extractUsage(translated)
				if pt2 != 0 || ct2 != 0 {
					pt, ct = pt2, ct2
					cost = h.costForModel(model, pt, ct)
				}
				outBody = translated
			}
		}

		needsResponsesShape := opts.translatedResponses && endpoint == "responses" && lastStatus == 200
		if needsResponsesShape {
			// Anthropic upstream: normalize the message response into chat
			// shape first, otherwise chatToResponsesJSON has nothing to work
			// with (the old path sent chat-shaped requests to Anthropic and
			// never handled the reply at all).
			if isAnthropicUpstream {
				if chatShape := anthropicToOpenAIChatResponse(outBody, model); chatShape != nil {
					outBody = chatShape
				}
			}
			if converted := chatToResponsesJSON(outBody, model); converted != nil {
				if pt2, ct2 := extractUsage(converted); pt2 != 0 || ct2 != 0 {
					pt, ct = pt2, ct2
					cost = h.costForModel(model, pt, ct)
				}
				outBody = converted
				// Chain support for the next turn: remember this turn's
				// full chat history under the issued response id.
				// Anthropic-shaped request bodies are skipped (different
				// message dialect than the chat history the next turn
				// would expand).
				if !isAnthropicUpstream {
					h.storeTranslatedTurn(opts.prevResponseID, keyPrefix, providerID, model, body, outBody)
				}
			} else if lastStatus == 200 {
				// Conversion failed (no choices[], content-filter shape, …).
				// Relaying the raw chat JSON would hand the /v1/responses
				// client a wrong-protocol 200 with no error signal; fail
				// honestly instead.
				log.Error().Str("model", model).Str("provider", providerID).Msg("responses translation produced no usable output; returning 502")
				httperr.Proxy(w, http.StatusBadGateway, "upstream returned an untranslatable response for the responses endpoint")
				h.logRequestExtended(keyPrefix, providerID, model, endpoint, http.StatusBadGateway, time.Since(start).Milliseconds(), pt, ct, cost, isStream)
				return attemptOutcome{committed: true, status: http.StatusBadGateway}
			}
		}

		// Legacy /v1/completions clients need text_completion shape
		// (choices[].text), never a raw Anthropic message body.
		if endpoint == "completions" && isAnthropicUpstream && lastStatus == 200 {
			if converted := anthropicToOpenAICompletion(outBody, model); converted != nil {
				if pt2, ct2 := extractUsage(converted); pt2 != 0 || ct2 != 0 {
					pt, ct = pt2, ct2
					cost = h.costForModel(model, pt, ct)
				}
				outBody = converted
			}
		}

		h.recordUsage(keyPrefix, r, pt+ct, cost)

		if cacheKey != "" && lastStatus >= 200 && lastStatus < 400 && len(outBody) <= 1<<20 {
			hdr := http.Header{}
			copyHeader(hdr, lastHdr)
			hdr.Set("Content-Type", "application/json")
			c.Set(cacheKey, outBody, lastStatus, hdr, cacheTTL)
		}
		copyHeader(w.Header(), lastHdr)
		applyFallbackHeader(w.Header())
		// Body-rewriting paths (anthropic→openai conversions, responses
		// translation) change payload size; a copied stale Content-Length makes
		// clients hang or truncate ("unexpected EOF"). Drop it when rewritten.
		if len(outBody) != len(lastBody) {
			w.Header().Del("Content-Length")
		}
		if cacheStatus == "bypass" {
			w.Header().Set("X-Cache", "BYPASS")
		} else {
			w.Header().Set("X-Cache", "MISS")
		}
		if (needsChatShape || needsResponsesShape) && lastStatus == 200 {
			w.Header().Set("Content-Type", "application/json")
		}
		_, _, nd := extractUsageDetail(outBody)
		meta := &logMeta{CacheStatus: cacheStatus, FinishReason: nd.FinishReason, CacheRead: nd.CacheRead, CacheWrite: nd.CacheWrite, Reasoning: nd.Reasoning, FallbackChain: marshalAttemptChain(opts.attempts)}
		h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, lastStatus, time.Since(start).Milliseconds(), pt, ct, cost, false, body, outBody, meta)
		w.WriteHeader(lastStatus)
		w.Write(outBody)
		return attemptOutcome{committed: true, status: lastStatus}
	}
}

// errEmptyUpstreamBody marks an upstream 200 that carried no body — a
// channel-death signature on new-api style upstreams, not a real completion.
var errEmptyUpstreamBody = errors.New("upstream returned 200 with empty body")

// retryableCode maps transport failures to a synthetic retry-polling code.
func retryableCode(status int) int {
	if status == 0 {
		return 503
	}
	return status
}

// injectStreamUsage adds stream_options.include_usage to an OpenAI-style
// streaming body when absent and configured. Failures fall back silently.
func injectStreamUsage(body []byte) ([]byte, bool) {
	var m map[string]interface{}
	if json.Unmarshal(body, &m) != nil {
		return body, false
	}
	if m["stream"] != true {
		return body, false
	}
	so, ok := m["stream_options"].(map[string]interface{})
	switch {
	case !ok:
		m["stream_options"] = map[string]interface{}{"include_usage": true}
	default:
		if v, exists := so["include_usage"]; exists {
			if b, isBool := v.(bool); isBool && b {
				return body, false
			}
		}
		so["include_usage"] = true
	}
	out, err := json.Marshal(m)
	if err != nil {
		return body, false
	}
	return out, true
}

// streamPumpResult carries what pumpStream harvested.
type streamPumpResult struct {
	bodySnapshot     []byte // sampled stream bytes (bounded) for debug bodies
	promptTokens     int
	completionTokens int
	midStreamFailure bool
	clientGone       bool
	capture          *streamCapture // assembled-text capture; nil when body logging is off
}

// streamCapture accumulates the assistant-visible content of a stream into a
// chat-shaped response body, so the usage log stores "what the client read"
// instead of a raw wall of SSE frames. Text and tool-call argument deltas are
// appended as frames arrive (dialect-aware), and the final usage frame
// contributes the rich token detail. The accumulator is size-bounded: once
// cap bytes of assembled content have been seen, further deltas are dropped
// (the raw sample still bounds separately).
type streamCapture struct {
	buf        bytes.Buffer
	cap        int
	detail     usageDetail
	toolID     string // in-flight tool call id (chat dialect)
	toolName   string // in-flight tool call name (chat dialect)
	toolOpen   bool   // a tool-call argument stream is in progress
	toolHeader bool   // the current tool card header was written
	dropped    bool   // cap hit; stop appending text
}

func newStreamCapture(capBytes int) *streamCapture {
	if capBytes <= 0 {
		capBytes = 8192
	}
	if capBytes > 64<<10 {
		capBytes = 64 << 10
	}
	return &streamCapture{cap: capBytes}
}

// observe feeds one parsed SSE event. m may be nil for non-JSON data frames.
func (sc *streamCapture) observe(ev sseEvent, m map[string]interface{}, anthropic bool) {
	if sc == nil {
		return
	}
	// Rich usage detail rides whichever frame carries a usage object; token
	// counts are max-merged because 0 is a non-observation, while a non-empty
	// finish reason always wins (later terminal frames are more specific).
	if m != nil {
		var usage map[string]interface{}
		if u, ok := m["usage"].(map[string]interface{}); ok {
			usage = u
		} else if resp, ok := m["response"].(map[string]interface{}); ok {
			// responses API response.completed shape
			if u, ok := resp["usage"].(map[string]interface{}); ok {
				usage = u
			}
		} else if msg, ok := m["message"].(map[string]interface{}); ok {
			// anthropic message_start shape
			if u, ok := msg["usage"].(map[string]interface{}); ok {
				usage = u
			}
		}
		if usage != nil {
			d := usageDetailFromMap(usage)
			if d.CacheRead > 0 {
				sc.detail.CacheRead = d.CacheRead
			}
			if d.CacheWrite > 0 {
				sc.detail.CacheWrite = d.CacheWrite
			}
			if d.Reasoning > 0 {
				sc.detail.Reasoning = d.Reasoning
			}
			if d.FinishReason != "" {
				sc.detail.FinishReason = d.FinishReason
			}
		}
	}
	if sc.dropped {
		return
	}
	if typ, _ := m["type"].(string); strings.HasPrefix(typ, "response.") {
		// OpenAI Responses API dialect (native pass-through streams).
		sc.observeResponses(m)
		return
	}
	switch {
	case anthropic:
		sc.observeAnthropic(ev, m)
	default:
		sc.observeChat(m)
	}
}

// observeResponses accumulates Responses-API stream events: output text
// deltas, function-call argument deltas, and the terminal completed frame
// (whose assembled output array is used verbatim when no deltas were seen).
func (sc *streamCapture) observeResponses(m map[string]interface{}) {
	typ, _ := m["type"].(string)
	switch typ {
	case "response.output_text_delta":
		if d, ok := m["delta"].(map[string]interface{}); ok {
			if t, ok := d["text"].(string); ok {
				sc.appendText(t)
			}
		}
	case "response.output_item.added":
		if item, ok := m["item"].(map[string]interface{}); ok && item["type"] == "function_call" {
			sc.flushTool()
			if n, ok := item["name"].(string); ok && n != "" {
				sc.toolName = n
			}
			sc.toolOpen = true
		}
	case "response.function_call_arguments.delta":
		if s, ok := m["delta"].(string); ok && s != "" {
			sc.toolOpen = true
			sc.appendToolArgs(s)
		}
	case "response.completed", "response.incomplete", "response.failed":
		resp, _ := m["response"].(map[string]interface{})
		if resp == nil {
			return
		}
		// Fallback assembly from the terminal output array (covers upstreams
		// that stream no text deltas).
		if sc.buf.Len() == 0 {
			if out, ok := resp["output"].([]interface{}); ok {
				for _, itemRaw := range out {
					item, _ := itemRaw.(map[string]interface{})
					if item == nil || item["type"] != "message" {
						continue
					}
					if parts, ok := item["content"].([]interface{}); ok {
						for _, pRaw := range parts {
							p, _ := pRaw.(map[string]interface{})
							if p != nil && (p["type"] == "output_text" || p["type"] == "text") {
								if t, ok := p["text"].(string); ok {
									sc.appendText(t)
								}
							}
						}
					}
				}
			}
		}
		if sc.detail.FinishReason == "" {
			if typ == "response.incomplete" {
				if det, ok := resp["incomplete_details"].(map[string]interface{}); ok {
					switch det["reason"] {
					case "max_output_tokens":
						sc.detail.FinishReason = "length"
					case "content_filter":
						sc.detail.FinishReason = "content_filter"
					}
				}
			}
			if sc.detail.FinishReason == "" {
				if st, ok := resp["status"].(string); ok {
					sc.detail.FinishReason = normalizeFinishReason(st)
				}
			}
		}
	}
}

// observeChat accumulates chat.completion.chunk content/tool deltas and
// finish_reason from choices[0].
func (sc *streamCapture) observeChat(m map[string]interface{}) {
	if m == nil {
		return
	}
	choices, ok := m["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return
	}
	c0, _ := choices[0].(map[string]interface{})
	if c0 == nil {
		return
	}
	if fr, ok := c0["finish_reason"]; ok {
		if s := normalizeFinishReason(fr); s != "" {
			sc.detail.FinishReason = s
		}
	}
	delta, _ := c0["delta"].(map[string]interface{})
	if delta == nil {
		return
	}
	if tc, ok := delta["content"].(string); ok {
		sc.appendText(tc)
	}
	if rc, ok := delta["reasoning_content"].(string); ok && rc != "" {
		// reasoning tokens-in-text from OpenAI-compatible upstreams: keep a
		// short bounded trace, clearly labeled
		sc.appendLabeled("thinking", rc)
	}
	if tcs, ok := delta["tool_calls"].([]interface{}); ok && len(tcs) > 0 {
		for _, tcRaw := range tcs {
			tc, _ := tcRaw.(map[string]interface{})
			if tc == nil {
				continue
			}
			fn, _ := tc["function"].(map[string]interface{})
			name, _ := fn["name"].(string)
			if id, ok := tc["id"].(string); ok && id != "" {
				sc.flushTool()
				sc.toolID, sc.toolName, sc.toolOpen = id, name, true
			} else if name != "" && !sc.toolOpen {
				sc.toolID, sc.toolName, sc.toolOpen = "", name, true
			}
			if args, ok := fn["arguments"].(string); ok {
				sc.appendToolArgs(args)
			}
		}
	}
}

// observeAnthropic accumulates content_block text / thinking / tool_use
// input_json_delta frames plus stop_reason from message_delta (where it rides
// inside delta.stop_reason).
func (sc *streamCapture) observeAnthropic(ev sseEvent, m map[string]interface{}) {
	if m == nil {
		return
	}
	if delta, ok := m["delta"].(map[string]interface{}); ok {
		if sr, ok := delta["stop_reason"]; ok {
			if s := normalizeFinishReason(sr); s != "" {
				sc.detail.FinishReason = s
			}
		}
		if t, ok := delta["text"].(string); ok {
			sc.appendText(t)
		}
		if t, ok := delta["thinking"].(string); ok && t != "" {
			sc.appendLabeled("thinking", t)
		}
		if pjson, ok := delta["partial_json"].(string); ok {
			sc.toolOpen = true
			sc.appendToolArgs(pjson)
		}
	}
	if cb, ok := m["content_block"].(map[string]interface{}); ok {
		// content_block_start: a tool_use block names the call up front.
		if name, ok := cb["name"].(string); ok && name != "" {
			sc.flushTool()
			sc.toolName, sc.toolOpen = name, true
		}
		if t, ok := cb["text"].(string); ok {
			sc.appendText(t)
		}
	}
	_ = ev
}

// appendText appends assistant text, flushing any open tool-call card first.
func (sc *streamCapture) appendText(s string) {
	if s == "" || sc.dropped {
		return
	}
	sc.flushTool()
	if sc.buf.Len()+len(s) > sc.cap {
		s = s[:max(0, sc.cap-sc.buf.Len())]
		sc.dropped = true
	}
	sc.buf.WriteString(s)
}

// appendLabeled appends a clearly-labeled non-answer trace (thinking blocks).
func (sc *streamCapture) appendLabeled(label, s string) {
	if s == "" || sc.dropped {
		return
	}
	sc.flushTool()
	inner := "<" + label + ">" + s + "</" + label + ">"
	if sc.buf.Len()+len(inner) > sc.cap {
		sc.dropped = true
		return
	}
	sc.buf.WriteString(inner)
}

// appendToolArgs accumulates tool-call argument JSON fragments.
func (sc *streamCapture) appendToolArgs(args string) {
	if args == "" || sc.dropped {
		return
	}
	if !sc.toolOpen {
		sc.toolOpen = true
	}
	if sc.buf.Len()+len(args) > sc.cap {
		sc.dropped = true
		return
	}
	if !sc.toolHeader {
		sc.buf.WriteString("\n[tool_call " + sc.toolName + "] ")
		sc.toolHeader = true
	}
	sc.buf.WriteString(args)
}

// flushTool closes an open tool-call card in the assembled buffer.
func (sc *streamCapture) flushTool() {
	if sc.toolOpen {
		sc.buf.WriteString("\n")
		sc.toolOpen = false
		sc.toolHeader = false
	}
}

// body returns the assembled chat-shaped response body, or nil when nothing
// was assembled (caller then falls back to the raw stream sample).
func (sc *streamCapture) body() []byte {
	if sc == nil || sc.buf.Len() == 0 {
		return nil
	}
	sc.flushTool()
	resp := map[string]interface{}{
		"id":     "stream-capture",
		"object": "chat.completion",
		"choices": []interface{}{map[string]interface{}{
			"index":         0,
			"message":       map[string]interface{}{"role": "assistant", "content": sc.buf.String()},
			"finish_reason": sc.detail.FinishReason,
		}},
	}
	out, err := json.Marshal(resp)
	if err != nil {
		return nil
	}
	return out
}

// pumpStream relays an SSE body to the client with:
//   - idle-chunk watchdog (no more hard ceiling killing long generations),
//   - framing-aware SSE parsing across TCP chunk boundaries,
//   - protocol-correct termination on ANY abnormal exit,
//   - honest downstream accounting (usage-so-far, real outcome status).
//
// streamCacheCollector accumulates the exact client-bound bytes of one
// streaming response so a clean, fully-drained success stream can be cached
// and replayed verbatim for identical requests. Bounded: a stream larger
// than max is never cached (dropped permanently, no partial replays).
type streamCacheCollector struct {
	key     string
	ttl     int
	max     int
	buf     []byte
	dropped bool
}

// add appends client-bound stream bytes (called after commit, so keepalive
// heartbeats written by the TTFB watchdog are never captured).
func (s *streamCacheCollector) add(p []byte) {
	if s == nil || s.dropped || len(p) == 0 {
		return
	}
	if len(s.buf)+len(p) > s.max {
		s.dropped = true
		s.buf = nil
		return
	}
	s.buf = append(s.buf, p...)
}

// commit stores the captured stream when it is complete and cacheable.
func (s *streamCacheCollector) commit(c cache.Cache, status int, contentType string) {
	if s == nil || c == nil || s.dropped || len(s.buf) == 0 || s.ttl <= 0 {
		return
	}
	if status < 200 || status >= 300 {
		return
	}
	if contentType == "" {
		contentType = "text/event-stream"
	}
	c.Set(s.key, s.buf, status, http.Header{"Content-Type": []string{contentType}}, s.ttl)
}

func (h *Handler) pumpStream(w http.ResponseWriter, r *http.Request, upstreamCtx context.Context, resp *http.Response, model, keyPrefix, providerID, endpoint string, start time.Time, isAnthropicUpstream bool, applyFallbackHeader func(http.Header), ttfb *ttfbController, streamCache *streamCacheCollector, cacheStatus string) streamPumpResult {
	res := streamPumpResult{}

	commit := func() {
		if ttfb.headersCommitted() {
			// Keepalive umbra: SSE headers + first keepalive frame already
			// flowed from the pre-commit watchdog. Only the heartbeat has to
			// stop — real stream bytes now take over the wire.
			ttfb.stop()
			return
		}
		copyHeader(w.Header(), resp.Header)
		applyFallbackHeader(w.Header())
		w.Header().Set("X-Cache", "MISS")
		w.WriteHeader(resp.StatusCode)
	}
	// First flusher probe AFTER commit: statusRecorder + chi wrappers all implement Flush.
	flusher, _ := w.(http.Flusher)

	idle := h.Timeouts.StreamIdle
	// A zero/negative idle timeout means "watchdog disabled". time.NewTimer(0)
	// fires immediately and Stop() cannot retract the already-queued value, so
	// the watchdog branch below would close the body and kill every stream
	// within the first inter-chunk gap. A nil channel blocks forever when
	// received from — exactly the semantics "disabled" needs.
	var watchdog *time.Timer
	var watchdogC <-chan time.Time
	if idle > 0 {
		watchdog = time.NewTimer(idle)
		watchdogC = watchdog.C
	}
	defer func() {
		if watchdog != nil {
			watchdog.Stop()
		}
	}()

	type chunkMsg struct {
		data [8192]byte
		n    int
		err  error
	}
	chunks := make(chan chunkMsg) // sized copies avoid aliasing
	go func() {
		for {
			var cm chunkMsg
			n, err := resp.Body.Read(cm.data[:])
			cm.n, cm.err = n, err
			select {
			case chunks <- cm:
			case <-upstreamCtx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()

	var pending []byte // partial SSE event carry-over between reads
	var promptTok, completeTok int
	// Usage detail (cache/reasoning/finish) harvested independently of body
	// logging: it rides the billing parse of every frame, so streams get
	// full cache-token reporting even with LOG_BODIES off. Previously this
	// detail only survived inside `capture`, which is allocated for body
	// logging — silently zeroing cache_read_tokens for all streams.
	var liveDetail usageDetail
	sample := &bytes.Buffer{}
	sampleCap := h.BodyLogMaxBytes
	if sampleCap > 32<<10 {
		sampleCap = 32 << 10
	}
	// Assembled-text capture: only allocated when body logging is on, so the
	// privacy-off path adds zero per-frame work beyond what billing needs.
	var capture *streamCapture
	if h.LogBodies {
		capture = newStreamCapture(h.BodyLogMaxBytes)
	}

	harvest := func(frame []byte) {
		events := parseSSEEvents(frame)
		for _, ev := range events {
			if sampleCap > 0 && sample.Len() < sampleCap {
				snapshot := make([]byte, 0, len(ev.data)+len(ev.name)+16)
				if ev.name != "" {
					snapshot = append(snapshot, []byte(ev.name+": ")...)
				}
				snapshot = append(snapshot, ev.data...)
				snapshot = append(snapshot, '\n')
				if sample.Len()+len(snapshot) > sampleCap {
					snapshot = snapshot[:sampleCap-sample.Len()]
				}
				sample.Write(snapshot)
			}
			if bytes.Equal(bytes.TrimSpace(ev.data), []byte("[DONE]")) {
				continue
			}
			var evMap map[string]interface{}
			if len(ev.data) > 0 && ev.data[0] == '{' {
				if err := json.Unmarshal(ev.data, &evMap); err != nil {
					evMap = nil
				}
			}
			if capture != nil {
				capture.observe(ev, evMap, isAnthropicUpstream)
			}
			// Billing tokens: one parse feeds both extractions. Only the
			// anthropic cache fields need folding — OpenAI's
			// prompt_tokens_details.cached_tokens is a BREAKDOWN of
			// prompt_tokens (already included), never additive.
			pt, ct, pd := extractUsageDetailFromMap(evMap)
			if pd.CacheRead > 0 {
				liveDetail.CacheRead = pd.CacheRead
			}
			if pd.CacheWrite > 0 {
				liveDetail.CacheWrite = pd.CacheWrite
			}
			if pd.Reasoning > 0 {
				liveDetail.Reasoning = pd.Reasoning
			}
			if pd.FinishReason != "" {
				liveDetail.FinishReason = pd.FinishReason
			}
			if pd.CacheRead > 0 || pd.CacheWrite > 0 {
				anthropicCache := pd.CacheWrite // cache_write is anthropic-only
				if usageMap, ok := evMap["usage"].(map[string]interface{}); ok {
					if _, hasFlatCache := usageMap["cache_read_input_tokens"]; hasFlatCache {
						anthropicCache += pd.CacheRead
					}
				}
				pt += anthropicCache
			}
			if pt > 0 {
				promptTok = pt
			}
			if ct > 0 {
				completeTok = ct
			}
			// Anthropic puts input tokens on message_start and output deltas on
			// message_delta; harvest wherever present.
			aPt, aCt := harvestAnthropicTokens(ev.data)
			if aPt > 0 {
				promptTok = aPt
			}
			if aCt > 0 {
				completeTok = aCt
			}
		}
	}

	// Terminators and reframing must speak the CLIENT's dialect, not the
	// upstream's: chat.completions/completions clients receive OpenAI frames
	// even when the upstream is an anthropic translation target.
	clientDialectAnthropic := isAnthropicUpstream && endpoint == "messages"
	// Reframe anthropic-dialect SSE into legacy OpenAI completion chunks for
	// /v1/completions callers routed onto an anthropic upstream (comp-D2).
	var reframe *anthToOpenAICompletionStream
	if isAnthropicUpstream && endpoint == "completions" {
		reframe = newAnthToOpenAICompletionStream(model)
	}

	fail := func(reason string, clientGone bool) streamPumpResult {
		res.midStreamFailure = true
		res.clientGone = clientGone
		res.promptTokens, res.completionTokens = promptTok, completeTok
		res.bodySnapshot = sample.Bytes()
		res.capture = capture
		cost := h.costForModel(model, promptTok, completeTok)
		logStatus := http.StatusBadGateway
		if clientGone {
			logStatus = 499 // nginx convention: client closed request
		}
		if !clientGone {
			writeStreamTerminator(w, flusher, clientDialectAnthropic, reason)
		}
		log.Error().Str("model", model).Str("provider", providerID).Str("reason", reason).Bool("client_gone", clientGone).Msg("mid-stream failure terminated")
		h.recordUsage(keyPrefix, r, promptTok+completeTok, cost)
		h.logRequestStreamed(keyPrefix, providerID, model, endpoint, logStatus, time.Since(start).Milliseconds(), promptTok, completeTok, cost, sample.Bytes(), capture, cacheStatus, liveDetail)
		return res
	}

	first := true
	finishClean := func() streamPumpResult {
		// Parse any trailing partial event buffered without a final newline.
		if len(pending) > 0 {
			harvest(pending)
			pending = nil
		}
		res.promptTokens, res.completionTokens = promptTok, completeTok
		res.bodySnapshot = sample.Bytes()
		res.capture = capture
		cost := h.costForModel(model, promptTok, completeTok)
		if reframe != nil {
			if tail := reframe.finish(); len(tail) > 0 {
				w.Write(tail)
				if flusher != nil {
					flusher.Flush()
				}
				streamCache.add(tail)
			}
		}
		// Clean, fully-drained success stream: make it replayable for the
		// next identical request (stream caching is opt-in; nil collector =
		// disabled, commit is a no-op).
		streamCache.commit(h.cacheOrNoop(), resp.StatusCode, resp.Header.Get("Content-Type"))
		h.recordUsage(keyPrefix, r, promptTok+completeTok, cost)
		h.logRequestStreamed(keyPrefix, providerID, model, endpoint, resp.StatusCode, time.Since(start).Milliseconds(), promptTok, completeTok, cost, sample.Bytes(), capture, cacheStatus, liveDetail)
		return res
	}

	for {
		select {
		case <-r.Context().Done():
			resp.Body.Close()
			drainChan(chunks)
			return fail("gateway client disconnected", true)

		case <-watchdogC:
			resp.Body.Close() // unblocks the pump goroutine
			drainChan(chunks)
			// watchdogC only receives when idle > 0 (nil channel never fires).
			return fail("upstream idle timeout: no data received within "+idle.String(), false)

		case cm := <-chunks:
			if watchdog != nil && ((idle > 0 && !first) || cm.n > 0) {
				watchdog.Reset(idle)
			}

			// Process payload FIRST: Go's Read may deliver n>0 together with
			// err==io.EOF on the same call.
			if cm.n > 0 {
				buf := cm.data[:cm.n]
				if first {
					// Pre-commit sanity: never present an HTML error splash (or any
					// non-SSE payload) as a "successful" event stream to SSE clients.
					ct := resp.Header.Get("Content-Type")
					trimmed := bytes.TrimLeft(buf, " \t\r\n")
					sseish := strings.HasPrefix(ct, "text/event-stream") ||
						bytes.HasPrefix(trimmed, []byte("data:")) ||
						bytes.HasPrefix(trimmed, []byte("event:")) ||
						bytes.HasPrefix(trimmed, []byte(":"))
					if !sseish {
						resp.Body.Close()
						drainChan(chunks)
						res.bodySnapshot = append(res.bodySnapshot[:0], buf...)
						log.Error().Str("model", model).Str("provider", providerID).
							Str("content_type", ct).Msg("upstream returned non-SSE payload for streaming request")
						h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint,
							http.StatusBadGateway, time.Since(start).Milliseconds(), 0, 0, 0, true, nil, buf)
						res.midStreamFailure = true
						if ttfb.headersCommitted() {
							// Keepalive headers already flowed; deliver the
							// verdict in-band instead of a second status line.
							ttfb.stop()
							writeSSEUpstreamError(w, sseClientDialect(endpoint, isAnthropicUpstream), "upstream returned a non-stream payload for a streaming request")
							return res
						}
						// Nothing committed: answer honestly instead of an
						// implicit empty 200.
						httperr.Proxy(w, http.StatusBadGateway, "upstream returned a non-stream payload for a streaming request")
						return res
					}
					first = false
					commit()
				}
				if reframe != nil {
					// Rewrite the upstream dialect into client chunks instead of
					// relaying raw anthropic frames.
					if out := reframe.consume(buf); len(out) > 0 {
						w.Write(out)
						if flusher != nil {
							flusher.Flush()
						}
						streamCache.add(out)
					}
					pending = append(pending, buf...) // raw kept for harvest/sample
				} else {
					w.Write(buf)
					if flusher != nil {
						flusher.Flush()
					}
					streamCache.add(buf)
					pending = append(pending, buf...)
				}
				if idx := bytes.LastIndexByte(pending, '\n'); idx >= 0 {
					frame := pending[:idx+1]
					harvest(frame)
					rest := append([]byte(nil), pending[idx+1:]...)
					pending = rest
				}
				if len(pending) > 1<<20 { // runaway non-SSE payload guard
					harvest(pending)
					pending = pending[:0]
				}
			}

			if cm.err != nil {
				if isCleanEOF(cm.err) {
					if first {
						// Zero bytes AND EOF before any data: upstream accepted
						// the connection then produced nothing usable.
						resp.Body.Close()
						drainChan(chunks)
						return fail("upstream produced no stream data", false)
					}
					resp.Body.Close()
					return finishClean()
				}
				resp.Body.Close()
				drainChan(chunks)
				return fail("upstream stream error: "+cm.err.Error(), false)
			}
		}
	}
}

func isCleanEOF(err error) bool {
	// ONLY a true io.EOF means the upstream finished its stream.
	// io.ErrUnexpectedEOF indicates cut framing; treating it as clean
	// terminated streams WITHOUT a protocol terminator and logged fake
	// successes (anth-D2).
	return err == io.EOF
}

// drainChan empties a channel without blocking so its producer goroutine can exit.
func drainChan[T any](ch <-chan T) {
	for {
		select {
		case _, ok := <-ch:
			if !ok {
				return
			}
		default:
			return
		}
	}
}

func isUnexpectedEOF(err error) bool {
	return err != nil && (err == io.ErrUnexpectedEOF || strings.Contains(err.Error(), "unexpected EOF"))
}

// harvestAnthropicTokens extracts anthropic usage fields wherever they appear
// inside a JSON event payload (message_start carries input_tokens,
// message_delta/output_tokens arrive cumulatively).
func harvestAnthropicTokens(data []byte) (prompt, completion int) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return 0, 0
	}
	var m map[string]interface{}
	if json.Unmarshal(trimmed, &m) != nil {
		return 0, 0
	}
	var find func(node interface{}, depth int)
	find = func(node interface{}, depth int) {
		if depth > 4 {
			return
		}
		obj, ok := node.(map[string]interface{})
		if !ok {
			return
		}
		if u, ok := obj["usage"].(map[string]interface{}); ok {
			if v, ok := u["input_tokens"]; ok && toInt(v) > prompt {
				prompt = toInt(v)
			}
			// Anthropic prompt-cache tokens: message_start /
			// message_delta carry them alongside input_tokens.
			cacheSum := 0
			if v, ok := u["cache_read_input_tokens"]; ok {
				cacheSum += toInt(v)
			}
			if v, ok := u["cache_creation_input_tokens"]; ok {
				cacheSum += toInt(v)
			}
			if cacheSum > 0 {
				prompt += cacheSum
			}
			if v, ok := u["output_tokens"]; ok && toInt(v) > completion {
				completion = toInt(v)
			}
		}
		for _, child := range obj {
			find(child, depth+1)
		}
	}
	find(m, 0)
	return
}

// contentToText extracts assistant text from a chat message content field,
// which legal chat upstreams may return either as a string OR as an array of
// typed parts ([{"type":"text","text":"..."}]). Previously only the string
// form was read and array content silently vanished from Responses output.
func contentToText(v interface{}) string {
	switch c := v.(type) {
	case string:
		return c
	case []interface{}:
		var b strings.Builder
		for _, part := range c {
			pm, ok := part.(map[string]interface{})
			if !ok {
				continue
			}
			switch pm["type"] {
			case "text", "output_text":
				if t, ok := pm["text"].(string); ok {
					b.WriteString(t)
				}
			}
		}
		return b.String()
	default:
		return ""
	}
}

// chatToResponsesJSON converts a translated chat.completion payload back into
// OpenAI Responses-API shape so /v1/responses clients always receive the
// protocol they asked for (streaming variants are refused earlier — wrong-
// shaped events are worse than a clear error).
func chatToResponsesJSON(body []byte, model string) []byte {
	var chat map[string]interface{}
	if json.Unmarshal(body, &chat) != nil {
		return nil
	}
	choices, ok := chat["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil
	}
	first, _ := choices[0].(map[string]interface{})
	msg, _ := first["message"].(map[string]interface{})
	text := contentToText(msg["content"])
	usage, _ := chat["usage"].(map[string]interface{})
	inTok, outTok := 0, 0
	if usage != nil {
		if v, ok := usage["prompt_tokens"]; ok {
			inTok = toInt(v)
		}
		if v, ok := usage["completion_tokens"]; ok {
			outTok = toInt(v)
		}
	}
	id, _ := chat["id"].(string)
	if id == "" {
		id = "resp-" + uuid.NewString()
	}
	finish, _ := first["finish_reason"].(string)
	status := "completed"
	incompleteReason := ""
	switch finish {
	case "", "stop", "tool_calls", "function_call":
		status = "completed"
	case "length":
		status = "incomplete"
		incompleteReason = "max_output_tokens"
	case "content_filter":
		status = "incomplete"
		incompleteReason = "content_filter"
	default:
		status = "incomplete"
		incompleteReason = finish
	}
	// Tool calls: a chat tool_call must become a Responses function_call
	// output item, or /v1/responses agents can never see the model's tool
	// invocations (the pre-fix behavior emitted an empty assistant message
	// and dropped the calls on the floor).
	output := []map[string]interface{}{{
		"type":   "message",
		"id":     "msg_" + uuid.NewString()[:8],
		"role":   "assistant",
		"status": status,
		"content": []map[string]interface{}{
			{"type": "output_text", "text": text, "annotations": []interface{}{}},
		},
	}}
	if tcs, ok := msg["tool_calls"].([]interface{}); ok {
		for _, tcRaw := range tcs {
			tc, ok := tcRaw.(map[string]interface{})
			if !ok {
				continue
			}
			fn, _ := tc["function"].(map[string]interface{})
			name, _ := fn["name"].(string)
			if name == "" {
				continue
			}
			args, _ := fn["arguments"].(string)
			callID, _ := tc["id"].(string)
			output = append(output, map[string]interface{}{
				"type":      "function_call",
				"id":        "fc_" + uuid.NewString()[:8],
				"call_id":   callID,
				"name":      name,
				"arguments": args,
				"status":    "completed",
			})
		}
	}
	respObj := map[string]interface{}{
		"id":          id,
		"object":      "response",
		"created_at":  time.Now().Unix(),
		"status":      status,
		"model":       model,
		"output":      output,
		"output_text": text,
		"usage": map[string]interface{}{
			"input_tokens":  inTok,
			"output_tokens": outTok,
			"total_tokens":  inTok + outTok,
		},
	}
	if incompleteReason != "" {
		// Spec-required truncation detail (max_output_tokens / content_filter)
		// so clients can distinguish why a response is incomplete.
		respObj["incomplete_details"] = map[string]interface{}{"reason": incompleteReason}
	}
	out, err := json.Marshal(respObj)
	if err != nil {
		return nil
	}
	return out
}

// hasResponsesOutput reports whether a native /v1/responses JSON body carries
// the output array strict clients require. Bodies without it relay as a
// serialization failure on the client, so the caller falls through to the
// translated path instead.
func hasResponsesOutput(body []byte) bool {
	var probe map[string]interface{}
	if json.Unmarshal(body, &probe) != nil {
		return false
	}
	out, ok := probe["output"]
	if !ok || out == nil {
		return false
	}
	return true
}

// recordUsage forwards real usage to the budget ledger sink when wired.
func (h *Handler) recordUsage(keyPrefix string, r *http.Request, tokens int, costUSD float64) {
	if h.Usage == nil || (tokens == 0 && costUSD == 0) {
		return
	}
	var orgPtr *string
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil && k.OrgID != nil {
		orgPtr = k.OrgID
	}
	h.Usage.RecordUsage(keyPrefix, orgPtr, tokens, costUSD, time.Now().UTC())
}

// logRequestExtended is preserved for legacy callers/tests.
func (h *Handler) logRequestExtended(keyPrefix, providerID, model, endpoint string, status int, latencyMs int64, promptTokens, completionTokens int, costUSD float64, isStream bool) {
	h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, status, latencyMs, promptTokens, completionTokens, costUSD, isStream, nil, nil)
}

// logRequestStreamed persists a completed streaming exchange: the assembled
// chat-shaped body wins over the raw SSE sample when body logging is on, and
// the harvested usage detail (finish reason, cache/reasoning split) is stored
// on the row.
func (h *Handler) logRequestStreamed(keyPrefix, providerID, model, endpoint string, status int, latencyMs int64, promptTokens, completionTokens int, costUSD float64, rawSample []byte, capture *streamCapture, cacheStatus string, liveDetail usageDetail) {
	// Merge the always-on harvest with the capture's own (body-logging path)
	// so enabling LOG_BODIES never reduces detail.
	d := liveDetail
	if capture != nil {
		if capture.detail.CacheRead > d.CacheRead {
			d.CacheRead = capture.detail.CacheRead
		}
		if capture.detail.CacheWrite > d.CacheWrite {
			d.CacheWrite = capture.detail.CacheWrite
		}
		if capture.detail.Reasoning > d.Reasoning {
			d.Reasoning = capture.detail.Reasoning
		}
		if d.FinishReason == "" {
			d.FinishReason = capture.detail.FinishReason
		}
	}
	meta := &logMeta{
		FinishReason: d.FinishReason,
		CacheRead:    d.CacheRead,
		CacheWrite:   d.CacheWrite,
		Reasoning:    d.Reasoning,
		CacheStatus:  cacheStatus,
	}
	if capture == nil {
		h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, status, latencyMs, promptTokens, completionTokens, costUSD, true, nil, rawSample, meta)
		return
	}
	h.logRequestExtendedDetail(keyPrefix, providerID, model, endpoint, status, latencyMs, promptTokens, completionTokens, costUSD, true, rawSample, capture.body(), meta)
}

// logMeta carries the extended per-request observability fields beyond the
// classic usage row. Nil metas keep the legacy 15-column insert shape.
type logMeta struct {
	FinishReason  string
	FallbackChain string // JSON [{provider_id,name,status}] — attempted providers before the final one
	CacheRead     int
	CacheWrite    int
	Reasoning     int
	CacheHit      bool   // served from the gateway response cache (X-Cache: HIT)
	CacheStatus   string // hit | miss | bypass — disposition against the response cache
}

// logRequestExtendedDetail is the full-fidelity insert used when captured
// detail exists. Response body preference: assembled capture > raw sample.
func (h *Handler) logRequestExtendedDetail(keyPrefix, providerID, model, endpoint string, status int, latencyMs int64, promptTokens, completionTokens int, costUSD float64, isStream bool, rawSample, assembledBody []byte, meta *logMeta) {
	if meta == nil {
		meta = &logMeta{}
	}
	respBody := assembledBody
	if len(respBody) == 0 {
		respBody = rawSample
	}
	h.logRequestExtendedBodies(keyPrefix, providerID, model, endpoint, status, latencyMs, promptTokens, completionTokens, costUSD, isStream, nil, respBody, meta)
}

// logRequestExtendedBodies persists a completed exchange. When handler body
// logging is enabled (opt-in), captured payloads are truncated and scrubbed of
// credential material before insertion. meta, when present, adds the extended
// usage-metadata columns (finish reason, cache/reasoning tokens, fallback
// chain); legacy callers omit it and keep the original row shape.
func (h *Handler) logRequestExtendedBodies(keyPrefix, providerID, model, endpoint string, status int, latencyMs int64, promptTokens, completionTokens int, costUSD float64, isStream bool, requestBody, responseBody []byte, meta ...*logMeta) {
	if h.Metrics != nil {
		h.Metrics.IncRequests(providerID, model, endpoint, status)
		h.Metrics.ObserveLatency(providerID, endpoint, time.Duration(latencyMs)*time.Millisecond)
	}
	if h.DB == nil {
		return
	}
	id := uuid.NewString()
	total := promptTokens + completionTokens

	var m *logMeta
	if len(meta) > 0 {
		m = meta[0]
	}
	if m == nil {
		m = &logMeta{}
	}

	reqBodyStr := ""
	respBodyStr := ""
	if h.LogBodies {
		cap_ := h.BodyLogMaxBytes
		if cap_ <= 0 {
			cap_ = 8192
		}
		if len(requestBody) > 0 {
			b := requestBody
			if len(b) > cap_ {
				b = b[:cap_]
			}
			reqBodyStr = ScrubSecrets(string(b))
		}
		if len(responseBody) > 0 {
			b := responseBody
			if len(b) > cap_ {
				b = b[:cap_]
			}
			respBodyStr = ScrubSecrets(string(b))
		}
	}
	// cache_hit is an INTEGER column on both dialects: bind an int, never a
	// Go bool — lib/pq will not cast boolean→integer, so a bool parameter
	// fails the whole INSERT on Postgres (SQLite tolerates it, which is why
	// this must be an explicit conversion).
	cacheHit := 0
	if m.CacheHit {
		cacheHit = 1
	}
	if _, err := h.DB.Exec(db.Q(`INSERT INTO request_logs(id,key_prefix,provider_id,model,endpoint,status,latency_ms,created_at,prompt_tokens,completion_tokens,total_tokens,cost_usd,is_stream,request_body,response_body,finish_reason,fallback_chain,cache_read_tokens,cache_write_tokens,reasoning_tokens,cache_hit,cache_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`),
		id, keyPrefix, providerID, model, endpoint, status, latencyMs, time.Now().UTC(), promptTokens, completionTokens, total, costUSD, isStream, nullIfEmpty(reqBodyStr), nullIfEmpty(respBodyStr),
		nullIfEmpty(m.FinishReason), nullIfEmpty(m.FallbackChain), m.CacheRead, m.CacheWrite, m.Reasoning, cacheHit, nullIfEmpty(m.CacheStatus)); err != nil {
		// The insert error was previously discarded, which let a dialect
		// mismatch silently drop EVERY request row on Postgres. Log loudly so
		// the next regression surfaces immediately.
		log.Error().Err(err).Str("model", model).Str("endpoint", endpoint).Int("status", status).Msg("request_logs insert failed")
	}
	// Per-key analytics attribution: stamp the owning gateway key's id. Runs
	// after the primary insert so a failure here never loses the request log.
	if keyPrefix != "" && h.DB != nil {
		if keyID := h.gatewayKeyID(keyPrefix); keyID != "" {
			h.DB.Exec(db.Q(`UPDATE request_logs SET key_id=? WHERE id=?`), keyID, id)
		}
	}
}

// gatewayKeyID resolves a gateway key prefix to its stable gateway_keys.id,
// with an unbounded-but-bounded-in-practice cache (one entry per key). The
// empty string is cached for misses (session virtual keys, unknown prefixes)
// so repeated traffic doesn't re-query the DB.
func (h *Handler) gatewayKeyID(prefix string) string {
	if prefix == "" || h == nil || h.DB == nil {
		return ""
	}
	h.keyIDMu.RLock()
	if id, ok := h.keyIDCache[prefix]; ok {
		h.keyIDMu.RUnlock()
		return id
	}
	h.keyIDMu.RUnlock()

	var id string
	_ = h.DB.QueryRow(db.Q(`SELECT id FROM gateway_keys WHERE prefix=? LIMIT 1`), prefix).Scan(&id)

	h.keyIDMu.Lock()
	if h.keyIDCache == nil {
		h.keyIDCache = make(map[string]string)
	}
	h.keyIDCache[prefix] = id // empty on miss; cached so misses don't thrash
	h.keyIDMu.Unlock()
	return id
}

func nullIfEmpty(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}

func extractAnthropicUsageFromSSEChunk(chunk []byte) (prompt, completion int) {
	// look for "input_tokens": X and "output_tokens": Y in chunk
	// naive json extract
	var tmp map[string]interface{}
	// try to find json objects in chunk
	// split by \n and parse each
	lines := bytes.Split(chunk, []byte("\n"))
	for _, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			line = bytes.TrimPrefix(line, []byte("data:"))
			line = bytes.TrimSpace(line)
		}
		if len(line) == 0 || bytes.Equal(line, []byte("[DONE]")) {
			continue
		}
		if !bytes.HasPrefix(line, []byte("{")) {
			continue
		}
		var m map[string]interface{}
		if json.Unmarshal(line, &m) == nil {
			if usage, ok := m["usage"].(map[string]interface{}); ok {
				if v, ok := usage["input_tokens"]; ok {
					prompt = toInt(v)
				}
				if v, ok := usage["output_tokens"]; ok {
					completion = toInt(v)
				}
				if prompt != 0 || completion != 0 {
					return
				}
			}
			// also check top-level for message_delta usage
			if v, ok := m["usage"]; ok {
				if um, ok := v.(map[string]interface{}); ok {
					if iv, ok := um["input_tokens"]; ok {
						prompt = toInt(iv)
					}
					if ov, ok := um["output_tokens"]; ok {
						completion = toInt(ov)
					}
				}
			}
		}
		// also try to unmarshal chunk as whole map
		_ = tmp
	}
	return
}

func anthropicToOpenAIChatResponse(body []byte, model string) []byte {
	var anth map[string]interface{}
	if err := json.Unmarshal(body, &anth); err != nil {
		return nil
	}
	// check if it's anthropic message (has type message and content array)
	if t, ok := anth["type"].(string); !ok || t != "message" {
		return nil
	}
	contentArr, ok := anth["content"].([]interface{})
	if !ok {
		return nil
	}
	var text string
	for _, c := range contentArr {
		if cm, ok := c.(map[string]interface{}); ok {
			if cm["type"] == "text" {
				if txt, ok := cm["text"].(string); ok {
					text += txt
				}
			} else if cm["type"] == "thinking" {
				// skip thinking
			}
		}
	}
	usage, _ := anth["usage"].(map[string]interface{})
	promptTokens := 0
	completionTokens := 0
	if usage != nil {
		if v, ok := usage["input_tokens"]; ok {
			promptTokens = toInt(v)
		}
		if v, ok := usage["output_tokens"]; ok {
			completionTokens = toInt(v)
		}
	}
	id, _ := anth["id"].(string)
	if id == "" {
		id = "chatcmpl-" + uuid.NewString()[:8]
	}
	finish := mapStopReason(anth["stop_reason"])
	openAIResp := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": text,
				},
				"finish_reason": finish,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	out, _ := json.Marshal(openAIResp)
	return out
}

// mapStopReason translates an Anthropic stop_reason value into the OpenAI
// finish_reason vocabulary. max_tokens → "length"; end_turn/stop_sequence and
// anything unrecognized → "stop".
func mapStopReason(v interface{}) string {
	switch s, _ := v.(string); s {
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// anthropicToOpenAICompletion converts a native Anthropic message body into
// the LEGACY OpenAI text-completion shape consumed by /v1/completions clients
// (choices[].text, object "text_completion") — the chat-completions converter
// above emits choices[].message which those clients cannot read.
func anthropicToOpenAICompletion(body []byte, model string) []byte {
	var anth map[string]interface{}
	if err := json.Unmarshal(body, &anth); err != nil {
		return nil
	}
	if t, ok := anth["type"].(string); !ok || t != "message" {
		return nil
	}
	contentArr, ok := anth["content"].([]interface{})
	if !ok {
		return nil
	}
	var text string
	for _, c := range contentArr {
		if cm, ok := c.(map[string]interface{}); ok && cm["type"] == "text" {
			if txt, ok := cm["text"].(string); ok {
				text += txt
			}
		}
	}
	promptTokens, completionTokens := 0, 0
	if usage, _ := anth["usage"].(map[string]interface{}); usage != nil {
		if v, ok := usage["input_tokens"]; ok {
			promptTokens = toInt(v)
		}
		if v, ok := usage["output_tokens"]; ok {
			completionTokens = toInt(v)
		}
	}
	id, _ := anth["id"].(string)
	if id == "" {
		id = "cmpl-" + uuid.NewString()[:8]
	}
	resp := map[string]interface{}{
		"id":      id,
		"object":  "text_completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"text":          text,
				"finish_reason": mapStopReason(anth["stop_reason"]),
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     promptTokens,
			"completion_tokens": completionTokens,
			"total_tokens":      promptTokens + completionTokens,
		},
	}
	out, _ := json.Marshal(resp)
	return out
}

func (h *Handler) costForModel(modelID string, prompt, completion int) float64 {
	if h.CatalogStore == nil || (prompt == 0 && completion == 0) {
		return 0
	}
	m, _, err := h.CatalogStore.FindBestMatch(modelID)
	if err != nil {
		// Unknown model (not in the models.dev snapshot — new releases,
		// private or aliased upstream names). Returning $0 makes real
		// upstream spend invisible to cost budgets. Fall back to
		// operator-configured default pricing when present:
		//   price_fallback_input_usd_per_1m / price_fallback_output_usd_per_1m
		// (settable via the dashboard Settings page).
		in := h.settingFloat("price_fallback_input_usd_per_1m")
		out := h.settingFloat("price_fallback_output_usd_per_1m")
		if in > 0 || out > 0 {
			return float64(prompt)/1_000_000*in + float64(completion)/1_000_000*out
		}
		return 0
	}
	return catalog.CostFor(m, prompt, completion)
}

// settingFloat reads a numeric system_config key ("" / unparseable → 0).
func (h *Handler) settingFloat(key string) float64 {
	if h.DB == nil {
		return 0
	}
	var v sql.NullString
	if err := h.DB.QueryRow(db.Q(`SELECT value FROM system_config WHERE key=?`), key).Scan(&v); err != nil || !v.Valid {
		return 0
	}
	f, _ := strconv.ParseFloat(strings.TrimSpace(v.String), 64)
	return f
}

// modelSupportsAttachment reports whether provider_models flags the model as
// attachment-capable (audio/file/image inputs). Used to skip the sanitizer's
// destructive attachment stripping: a legal multimodal request to a capable
// model must reach the model with its attachments intact, not replaced by
// "[audio attachment omitted]" placeholders.
func (h *Handler) modelSupportsAttachment(providerID, modelID string) bool {
	if h.DB == nil {
		return false
	}
	var att sql.NullBool
	if err := h.DB.QueryRow(db.Q(`SELECT attachment FROM provider_models WHERE provider_id=? AND model_id=?`), providerID, modelID).Scan(&att); err != nil || !att.Valid {
		return false
	}
	return att.Bool
}

func (h *Handler) getReasoningConfig(providerID, modelID string) (reasoning bool, rType string, levels []string, limits map[string]int) {
	known, reasoning, rType, levels, limits := h.knownReasoningConfig(providerID, modelID)
	if !known {
		return false, "", nil, map[string]int{}
	}
	return reasoning, rType, levels, limits
}

// knownReasoningConfig resolves reasoning metadata and reports whether the
// model is KNOWN to the gateway at all (provider_models row or catalog
// entry). known=false means "no metadata" — distinct from known=true with
// reasoning=false ("metadata says this model cannot reason"). The distinction
// matters: validateReasoning refuses legal reasoning params on unknown
// models, which is a deployment-dependent false rejection.
func (h *Handler) knownReasoningConfig(providerID, modelID string) (known, reasoning bool, rType string, levels []string, limits map[string]int) {
	limits = map[string]int{}
	if h.DB == nil {
		return false, false, "", nil, limits
	}
	var r bool
	var rt, rl, rol sql.NullString
	pmQuery := db.Q(`SELECT reasoning, reasoning_type, reasoning_levels, reasoning_output_limits FROM provider_models WHERE provider_id=? AND model_id=?`)
	err := h.DB.QueryRow(pmQuery, providerID, modelID).Scan(&r, &rt, &rl, &rol)
	if err != nil {
		// /v1/models advertises qualified ids ("provider/model") but rows are
		// stored unqualified. Retry with the suffix; the lookup is already
		// scoped to the pinned provider, so stripping is safe.
		if short := shortModelID(modelID); short != modelID {
			err = h.DB.QueryRow(pmQuery, providerID, short).Scan(&r, &rt, &rl, &rol)
		}
	}
	if err != nil {
		// Devin stores one collapsed base row ("swe-2") while clients may
		// request explicit level variants ("devin/swe-2-max"): resolve to
		// the base row so effort validation and level metadata apply.
		if short, base := shortModelID(modelID), devin.CollapseReasoningVariant(shortModelID(modelID)); base != short {
			err = h.DB.QueryRow(pmQuery, providerID, base).Scan(&r, &rt, &rl, &rol)
		}
	}
	if err != nil {
		// fallback to catalog
		if h.CatalogStore != nil {
			if cm, cerr := h.CatalogStore.Get(modelID); cerr == nil {
				r = cm.Reasoning
				rt.String, rt.Valid = cm.ReasoningType, cm.ReasoningType != ""
				rl.String, rl.Valid = cm.ReasoningLevels, cm.ReasoningLevels != ""
				rol.String, rol.Valid = cm.ReasoningOutputLimits, cm.ReasoningOutputLimits != ""
				return true, r, rType, levels, limits
			} else if cm, cerr := h.CatalogStore.GetByShortID(shortModelID(modelID)); cerr == nil {
				r = cm.Reasoning
				rt.String, rt.Valid = cm.ReasoningType, cm.ReasoningType != ""
				rl.String, rl.Valid = cm.ReasoningLevels, cm.ReasoningLevels != ""
				rol.String, rol.Valid = cm.ReasoningOutputLimits, cm.ReasoningOutputLimits != ""
				return true, r, rType, levels, limits
			}
		}
		// No provider_models row and no catalog entry: the model is unknown.
		return false, false, "", nil, limits
	}
	if rt.Valid {
		rType = rt.String
	}
	if rl.Valid && rl.String != "" {
		json.Unmarshal([]byte(rl.String), &levels)
	}
	if rol.Valid && rol.String != "" {
		json.Unmarshal([]byte(rol.String), &limits)
	}
	return true, r, rType, levels, limits
}

func (h *Handler) validateReasoning(providerID, modelID string, body []byte) error {
	effort := translate.ExtractReasoningEffort(body)
	if effort == "" {
		return nil
	}
	effort = strings.ToLower(strings.TrimSpace(effort))
	known, reasoning, rType, levels, limits := h.knownReasoningConfig(providerID, modelID)
	// The gateway only refuses a reasoning request when it has POSITIVE
	// evidence the model cannot do it. A bare provider_models row with
	// reasoning=0 comes straight from discovery (models.dev enrichment
	// never ran) and means "nothing known", not "cannot reason" — the old
	// behavior turned that metadata gap into a client-visible 400 for legal
	// reasoning_effort/thinking requests, deployment-dependently. Unknown
	// models pass through; the upstream, which knows its own models,
	// validates.
	if !known {
		return nil
	}
	if len(levels) == 0 && !reasoning {
		// reasoning=0 with no configured level list is discovery-default
		// noise, not a positive capability claim — don't gate on it.
		return nil
	}
	if !reasoning {
		return fmt.Errorf("model %s does not support reasoning (effort %s requested)", modelID, effort)
	}
	if rType == "none" {
		return fmt.Errorf("model %s reasoning disabled", modelID)
	}
	if len(levels) > 0 {
		allowed := map[string]bool{}
		for _, lv := range levels {
			allowed[strings.ToLower(lv)] = true
		}
		if !allowed[effort] {
			return fmt.Errorf("reasoning effort '%s' not supported for model %s; allowed: %v", effort, modelID, levels)
		}
		// per-level output limit
		if limit, ok := limits[effort]; ok && limit > 0 {
			// extract max_tokens / max_output_tokens
			var tmp struct {
				MaxTokens       *int `json:"max_tokens"`
				MaxOutputTokens *int `json:"max_output_tokens"`
				MaxTokensAlt    *int `json:"max_tokens_alt"`
			}
			json.Unmarshal(body, &tmp)
			maxOut := 0
			if tmp.MaxTokens != nil {
				maxOut = *tmp.MaxTokens
			} else if tmp.MaxOutputTokens != nil {
				maxOut = *tmp.MaxOutputTokens
			}
			if maxOut > 0 && maxOut > limit {
				return fmt.Errorf("max_tokens %d exceeds output limit %d for reasoning effort '%s' on model %s", maxOut, limit, effort, modelID)
			}
		}
	}
	return nil
}

func (h *Handler) ChatCompletions(w http.ResponseWriter, r *http.Request) {
	body, bodyOK := h.readProxyBody(w, r)
	if !bodyOK {
		return
	}
	rawModel := translate.ExtractModel(body)
	model := h.resolveAlias(rawModel)
	if !h.enforceModelAllowlist(w, r, rawModel, model) {
		return
	}
	if !h.checkTPM(w, r, body) {
		return
	}
	if !chatMessagesPresent(body) {
		httperr.Invalid(w, "messages: required non-empty array of chat messages")
		return
	}
	// Spec-typed param validation: reject wrong-typed sampling/limit params
	// locally with a 400 naming the param. ckff.dev (new-api) answers these
	// with HTTP 500 "cannot unmarshal number 2.5 into field n of type int" —
	// a retryable-looking 5xx for a deterministic client mistake. OpenAI
	// itself 400s ("Invalid type for 'n': expected an integer").
	if err := validateChatParamTypes(body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	isStream := translate.IsStreaming(body)
	if model != rawModel {
		body = replaceModelInBody(body, model)
	}
	providerHint := r.Header.Get("X-Provider")
	candidates, rule := h.candidateProvidersWithRule(rawModel, model, providerHint, h.requestKeyOrg(r), func(p *models.Provider) bool {
		return p.Type != models.ProviderAnthropic
	})
	if len(candidates) == 0 {
		if h.unroutedModel() {
			noRouteFor(w, model)
			return
		}
		if p, err := h.ProviderStore.Resolve(model, providerHint); err == nil && p != nil && p.Type == models.ProviderAnthropic {
			httperr.Invalid(w, "model '"+model+"' is an anthropic model; use POST /v1/messages instead of /v1/chat/completions")
			return
		}
		httperr.Proxy(w, http.StatusServiceUnavailable, "no provider configured")
		return
	}
	if err := h.validateReasoning(candidates[0].Provider.ID, model, body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	// Multi-protocol providers serve each model on exactly one upstream
	// endpoint. Fail fast with the right endpoint instead of forwarding a
	// chat body to a messages/responses-only model for a cryptic 400.
	if p0 := candidates[0].Provider; isMultiProtocolProvider(p0) {
		if api := UpstreamAPIForModel(p0, model); api == UpstreamMessages || api == UpstreamResponses {
			httperr.Invalid(w, "model '"+model+"' is served via "+correctInboundFor(api)+" on this provider; use POST "+correctInboundFor(api)+" instead of /v1/chat/completions")
			return
		}
	}
	start := time.Now()
	keyPrefix := r.Header.Get("X-Gateway-Key-Prefix")
	// Antigravity speaks Cloud Code Assist (OAuth), not OpenAI — translate.
	if p0 := candidates[0].Provider; isAntigravity(p0) {
		h.proxyAntigravity(w, r, body, body, isStream, model, "chat.completions", keyPrefix, start, p0)
		return
	}
	// Devin speaks Connect-proto (OAuth session token), not OpenAI — translate.
	if p0 := candidates[0].Provider; isDevin(p0) {
		h.proxyDevin(w, r, body, isStream, model, "chat.completions", keyPrefix, start, p0)
		return
	}
	h.proxyCandidates(w, r, body, isStream, model, "chat.completions", keyPrefix, start, candidates, rule, func(p *models.Provider, body []byte) (string, string, []byte, bool, error) {
		apiKey, err := h.ProviderStore.DecryptKey(p)
		if err != nil {
			return "", "", nil, false, err
		}
		// The body may already carry a rule member's model override; derive
		// the upstream id from it so the override survives prefix stripping.
		bodyModel := translate.ExtractModel(body)
		upstream := stripProviderPrefix(bodyModel, p)
		out := body
		if upstream != bodyModel {
			out = replaceModelInBody(body, upstream)
		}
		// Strict OpenAI-compatible upstreams reject legal-but-sloppy shapes
		// (tool msgs without tool_call_id, role:"developer", assistant
		// content:null) with 400 "Invalid input". Normalize before sending;
		// no-op on bodies it does not understand. Attachment-capable models
		// keep their audio/file parts verbatim.
		out = sanitizeOpenAICompatBodyOpts(out, sanitizeOpts{keepAttachments: h.modelSupportsAttachment(p.ID, upstream)})
		// Billing accuracy: ask the upstream for a final usage frame when the
		// client didn't (OpenAI-compat endpoints only; guarded per-dialect).
		if isStream && h.StreamUsageInject {
			if b2, changed := injectStreamUsage(out); changed {
				out = b2
			}
		}
		target := upstreamTarget(p.BaseURL, "/chat/completions", model, p.Type == models.ProviderAzure)
		return target, apiKey, out, false, nil
	})
}

// Completions handles POST /v1/completions

// Completions handles POST /v1/completions
func (h *Handler) Completions(w http.ResponseWriter, r *http.Request) {
	body, bodyOK := h.readProxyBody(w, r)
	if !bodyOK {
		return
	}
	rawModel := translate.ExtractModel(body)
	model := h.resolveAlias(rawModel)
	if !h.enforceModelAllowlist(w, r, rawModel, model) {
		return
	}
	if !h.checkTPM(w, r, body) {
		return
	}
	if model != rawModel {
		body = replaceModelInBody(body, model)
	}
	isStream := translate.IsStreaming(body)
	providerHint := r.Header.Get("X-Provider")
	candidates, _ := h.candidateProvidersWithRule(rawModel, model, providerHint, h.requestKeyOrg(r), nil)
	if len(candidates) == 0 {
		if h.unroutedModel() {
			noRouteFor(w, model)
			return
		}
		httperr.Proxy(w, http.StatusServiceUnavailable, "no provider configured")
		return
	}
	p := candidates[0].Provider
	// Per-member model override from a curated rule (rule-routed traffic only).
	if override := candidates[0].ModelOverrideForRequest(); override != "" && override != model {
		body = replaceModelInBody(body, override)
		model = override
	}
	if err := h.validateReasoning(p.ID, model, body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	if p.Type == models.ProviderAnthropic {
		httperr.Invalid(w, "model '"+model+"' is an anthropic model; use POST /v1/messages instead of /v1/completions")
		return
	}
	// Multi-protocol providers serve messages/responses models on their own
	// endpoints; legacy /v1/completions cannot speak them.
	if isMultiProtocolProvider(p) {
		if api := UpstreamAPIForModel(p, model); api == UpstreamMessages || api == UpstreamResponses {
			httperr.Invalid(w, "model '"+model+"' is served via "+correctInboundFor(api)+" on this provider; use POST "+correctInboundFor(api)+" instead of /v1/completions")
			return
		}
	}
	apiKey, derr := h.ProviderStore.DecryptKey(p)
	if derr != nil {
		// Fail fast (502) instead of sending "Bearer " + "" upstream to get a
		// confusing 401 from the provider.
		httperr.Write(w, http.StatusBadGateway, "provider credential unavailable", httperr.TypeProxy)
		return
	}
	target := upstreamTarget(p.BaseURL, "/completions", model, p.Type == models.ProviderAzure)
	isAnthropic := p.Type == models.ProviderAnthropic || strings.Contains(strings.ToLower(p.BaseURL), "anthropic") || strings.Contains(strings.ToLower(p.Name), "claude") || strings.Contains(strings.ToLower(model), "claude") || strings.Contains(strings.ToLower(model), "muse-spark")
	if !isAnthropic {
		// Strip the "provider/model" prefix the client used for routing:
		// chat/embeddings do this; completions forwarded "ckff/glm-5.3-flash"
		// verbatim and upstreams answered 503 "No available channel for model
		// ckff/…" — a deterministic routing error.
		upstream := stripProviderPrefix(model, p)
		if upstream != translate.ExtractModel(body) {
			body = replaceModelInBody(body, upstream)
		}
	}
	if isAnthropic {
		var comp map[string]any
		if json.Unmarshal(body, &comp) == nil {
			var chatBody map[string]any
			if prompt, hasPrompt := comp["prompt"]; hasPrompt && prompt != nil {
				var promptStr string
				switch v := prompt.(type) {
				case string:
					promptStr = v
				case []interface{}:
					// Legal legacy shapes: string arrays (multiple prompts)
					// and token-id arrays. Multiple prompts cannot map onto a
					// single Anthropic completion — keep every string (joined)
					// rather than silently dropping all but the first; token
					// arrays are refused honestly below instead of being
					// forwarded as literal junk text.
					strs := make([]string, 0, len(v))
					allStr := len(v) > 0
					for _, item := range v {
						s, ok := item.(string)
						if !ok {
							allStr = false
							break
						}
						strs = append(strs, s)
					}
					if allStr {
						promptStr = strings.Join(strs, "\n\n")
					}
				default:
					b, _ := json.Marshal(prompt)
					if len(b) > 2 && b[0] == '"' {
						var s string
						json.Unmarshal(b, &s)
						promptStr = s
					} else {
						promptStr = string(b)
					}
				}
				if promptStr == "" || promptStr == "null" {
					// Unusable prompt (empty, null, or token-id array).
					// The old behavior fabricated the prompt "hi" and billed
					// the client for a completion they never asked for.
					httperr.Invalid(w, "prompt: a string prompt is required for anthropic-backed models (token-id arrays are not supported)")
					return
				}
				chatBody = map[string]any{
					"model":    comp["model"],
					"messages": []map[string]any{{"role": "user", "content": promptStr}},
				}
				// Carry the standard sampling/cap params through the chat
				// translation (OpenAIToAnthropic maps them); previously only
				// max_tokens/max_output_tokens/stream survived, so e.g.
				// temperature:0 silently ran at provider default.
				for _, k := range []string{"max_tokens", "max_output_tokens", "temperature", "top_p", "stop", "seed", "frequency_penalty", "presence_penalty"} {
					if v, ok := comp[k]; ok && v != nil {
						chatBody[k] = v
					}
				}
				if v, ok := comp["stream"]; ok {
					chatBody["stream"] = v
				}
			} else if _, hasMsgs := comp["messages"]; hasMsgs {
				// A messages-shaped body routed through /v1/completions:
				// validate instead of fabricating. messages:null previously
				// produced a fake user turn "hi"; nil-content messages were
				// silently rewritten too.
				msgs, ok := comp["messages"].([]interface{})
				if !ok || len(msgs) == 0 {
					httperr.Invalid(w, "messages: a non-empty messages array is required")
					return
				}
				for _, m := range msgs {
					mm, ok := m.(map[string]interface{})
					if !ok || mm["content"] == nil {
						httperr.Invalid(w, "messages: every message requires content")
						return
					}
				}
				chatBody = comp
			}
			if chatBody != nil {
				if b, err := json.Marshal(chatBody); err == nil {
					body = b
				}
			}
		}
		translated, _, err := translate.OpenAIToAnthropic(body)
		if err == nil {
			body = translated
			target = strings.TrimRight(p.BaseURL, "/") + "/v1/messages"
		}
	}
	start := time.Now()
	out := h.proxyWithMetrics(w, r, target, apiKey, body, isStream, model, p.ID, r.Header.Get("X-Gateway-Key-Prefix"), "completions", start, isAnthropic)
	if !out.committed {
		// Client-caused upstream 5xx: relay the upstream's verdict (the
		// generic "upstream unavailable" envelope would misclassify a
		// deterministic client error as a provider health issue).
		if out.clientCaused && out.errSnippet != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(outboundStatus(out.status))
			_, _ = w.Write([]byte(out.errSnippet))
			return
		}
		// Pre-commit terminal failure: write an explicit error instead of
		// letting net/http emit an implicit empty 200.
		httperr.Proxy(w, outboundStatus(out.status), "upstream unavailable")
	}
}

func (h *Handler) Embeddings(w http.ResponseWriter, r *http.Request) {
	body, bodyOK := h.readProxyBody(w, r)
	if !bodyOK {
		return
	}
	rawModel := translate.ExtractModel(body)
	model := h.resolveAlias(rawModel)
	if !h.enforceModelAllowlist(w, r, rawModel, model) {
		return
	}
	if !h.checkTPM(w, r, body) {
		return
	}
	if model != rawModel {
		body = replaceModelInBody(body, model)
	}
	providerHint := r.Header.Get("X-Provider")
	candidates, rule := h.candidateProvidersWithRule(rawModel, model, providerHint, h.requestKeyOrg(r), func(p *models.Provider) bool {
		return p.Type != models.ProviderAnthropic
	})
	if len(candidates) == 0 {
		if h.unroutedModel() {
			noRouteFor(w, model)
			return
		}
		httperr.Proxy(w, http.StatusServiceUnavailable, "no provider configured")
		return
	}
	start := time.Now()
	h.proxyCandidates(w, r, body, false, model, "embeddings", r.Header.Get("X-Gateway-Key-Prefix"), start, candidates, rule, func(p *models.Provider, body []byte) (string, string, []byte, bool, error) {
		apiKey, err := h.ProviderStore.DecryptKey(p)
		if err != nil {
			return "", "", nil, false, err
		}
		// The body may already carry a rule member's model override; derive
		// the upstream id from it so the override survives prefix stripping.
		bodyModel := translate.ExtractModel(body)
		upstream := stripProviderPrefix(bodyModel, p)
		out := body
		if upstream != bodyModel {
			out = replaceModelInBody(body, upstream)
		}
		target := upstreamTarget(p.BaseURL, "/embeddings", model, p.Type == models.ProviderAzure)
		return target, apiKey, out, false, nil
	})
}

// GetModel serves GET /v1/models/{id} — OpenAI SDK "retrieve model" parity.
// The route previously did not exist and fell through to the SPA 404.
func (h *Handler) GetModel(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	if id == "" {
		httperr.Invalid(w, "model id required")
		return
	}
	// Enforce per-key allowlists the same way the list endpoint does.
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil && len(k.AllowedModels) > 0 {
		if !apikey.IsModelAllowed(k.AllowedModels, id) {
			httperr.NotFound(w, "model not found")
			return
		}
	}
	if h.DB != nil {
		// Escape LIKE wildcards so ids like "%", "g%" or "model_1" match
		// literally instead of acting as patterns (bound parameter, so this
		// is wrong-entity prevention, not injection).
		likeID := strings.NewReplacer(`\`, `\\`, `%`, `\%`, `_`, `\_`).Replace(id)
		var modelID, displayName, ownedBy sql.NullString
		err := h.DB.QueryRow(db.Q(`SELECT pm.model_id, COALESCE(pm.display_name,''), COALESCE(pm.owned_by,'system') FROM provider_models pm WHERE pm.model_id=? OR pm.model_id LIKE '%/'||? ESCAPE '\' LIMIT 1`), id, likeID).Scan(&modelID, &displayName, &ownedBy)
		if err == nil && modelID.Valid {
			owner := "system"
			if ownedBy.Valid && ownedBy.String != "" {
				owner = ownedBy.String
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"object": "model", "id": id, "owned_by": owner, "created": gatewayEpoch})
			return
		}
	}
	if h.CatalogStore != nil {
		if m, err := h.CatalogStore.Get(id); err == nil {
			owned := m.Provider
			if owned == "" {
				owned = "system"
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"object": "model", "id": id, "owned_by": owned, "created": gatewayEpoch})
			return
		}
	}
	httperr.NotFound(w, "model not found")
}

// gatewayEpoch is a process-stable "created" timestamp for gateway-emitted
// model objects. The real OpenAI API always includes created (unix seconds)
// and openai-python's Model type requires it — without it,
// client.models.list() fails pydantic validation. A fixed per-process value
// keeps cached/uncached list responses consistent.
var gatewayEpoch = time.Now().Unix()

// Models handles GET /v1/models - now returns provider_models (discovered per provider) enriched, not full catalog
func (h *Handler) Models(w http.ResponseWriter, r *http.Request) {
	ck := modelsCacheKey(r)
	keyPrefix := r.Header.Get("X-Gateway-Key-Prefix")
	if cached, status, headers, ok := h.cacheOrNoop().Get(ck); ok {
		h.serveCacheHit(w, cached, status, headers)
		h.logRequestExtendedBodies(keyPrefix, "", "", "models", status, 0, 0, 0, 0, false, nil, nil, &logMeta{CacheHit: true, CacheStatus: "hit"})
		return
	}
	// allowlist filter for Models — if key is restricted, only return allowed models
	var allowed []string
	if k, ok := middleware.GatewayKeyFromContext(r.Context()); ok && k != nil {
		allowed = k.AllowedModels
	}
	filterAllowed := func(list []map[string]interface{}) []map[string]interface{} {
		if len(allowed) == 0 {
			return list
		}
		var out []map[string]interface{}
		for _, m := range list {
			id, _ := m["id"].(string)
			short, _ := m["model_id"].(string)
			tgt, _ := m["alias_target"].(string)
			if apikey.IsModelAllowed(allowed, id) || apikey.IsModelAllowed(allowed, short) || (tgt != "" && apikey.IsModelAllowed(allowed, tgt)) {
				out = append(out, m)
			}
		}
		return out
	}
	providers, err := h.ProviderStore.List()
	// Try provider_models first - this is the dynamic per-provider list user wants
	if h.DB != nil {
		rows, err := h.DB.Query(db.Q(`SELECT pm.model_id, pm.display_name, pm.owned_by, pm.context_window, pm.max_output, pm.input_cost, pm.output_cost, pm.reasoning, pm.tool_call, pm.attachment, p.name, pm.reasoning_type, pm.reasoning_levels, pm.reasoning_output_limits FROM provider_models pm JOIN providers p ON p.id = pm.provider_id ORDER BY p.name, pm.model_id LIMIT 500`))
		if err == nil {
			defer rows.Close()
			var pmModels []map[string]interface{}
			for rows.Next() {
				var modelID, displayName, ownedBy, providerName string
				var ctx, maxOut sql.NullInt64
				var inCost, outCost sql.NullFloat64
				var reasoning, toolCall, attachment sql.NullBool
				var rType, rLevels, rLimits sql.NullString
				if err := rows.Scan(&modelID, &displayName, &ownedBy, &ctx, &maxOut, &inCost, &outCost, &reasoning, &toolCall, &attachment, &providerName, &rType, &rLevels, &rLimits); err == nil {
					qid := qualifiedModelID(providerName, modelID)
					m := map[string]interface{}{
						"id":       qid,
						"model_id": modelID,
						"object":   "model",
						"owned_by": providerName,
						"created":  gatewayEpoch,
					}
					if displayName != "" {
						m["display_name"] = displayName
					}
					if ctx.Valid {
						m["context_window"] = ctx.Int64
					}
					if maxOut.Valid {
						m["max_output"] = maxOut.Int64
					}
					if inCost.Valid {
						m["input_cost"] = inCost.Float64
					}
					if outCost.Valid {
						m["output_cost"] = outCost.Float64
					}
					if reasoning.Valid {
						m["reasoning"] = reasoning.Bool
					}
					if toolCall.Valid {
						m["tool_call"] = toolCall.Bool
					}
					if attachment.Valid {
						m["attachment"] = attachment.Bool
					}
					if rType.Valid {
						m["reasoning_type"] = rType.String
					}
					if rLevels.Valid {
						m["reasoning_levels"] = rLevels.String
					}
					if rLimits.Valid {
						m["reasoning_output_limits"] = rLimits.String
					}
					pmModels = append(pmModels, m)
				}
			}
			if len(pmModels) > 0 {
				// add aliases
				seen := map[string]bool{}
				for _, m := range pmModels {
					seen[m["id"].(string)] = true
				}
				if h.DB != nil {
					rows2, _ := h.DB.Query(db.Q(`SELECT alias, target FROM model_aliases`))
					if rows2 != nil {
						defer rows2.Close()
						for rows2.Next() {
							var alias, target string
							if err := rows2.Scan(&alias, &target); err != nil || alias == "" {
								// Never emit a nameless model row into the list.
								continue
							}
							if !seen[alias] {
								pmModels = append(pmModels, map[string]interface{}{"id": alias, "object": "model", "owned_by": "gateway-alias", "alias_target": target, "created": gatewayEpoch})
							}
						}
					}
				}
				h.writeJSONCached(w, r, ck, 30, map[string]interface{}{"object": "list", "data": filterAllowed(pmModels)})
				return
			}
		}
	}
	// Fallback: if no provider_models yet, try old behavior (provider live + catalog) but don't expose full catalog if no providers
	var catalogModels []map[string]interface{}
	if h.CatalogStore != nil && len(providers) > 0 {
		list, _ := h.CatalogStore.List("", "", false, 200, 0)
		for _, cm := range list {
			catalogModels = append(catalogModels, map[string]interface{}{
				"id":             qualifiedModelID(cm.Provider, cm.ID),
				"model_id":       shortModelID(cm.ID),
				"object":         "model",
				"owned_by":       cm.Provider,
				"created":        gatewayEpoch,
				"display_name":   cm.Name,
				"context_window": cm.ContextWindow,
				"max_output":     cm.MaxOutput,
				"input_cost":     cm.InputCost,
				"output_cost":    cm.OutputCost,
				"reasoning":      cm.Reasoning,
				"tool_call":      cm.ToolCall,
				"attachment":     cm.Attachment,
			})
		}
	}
	if err != nil || len(providers) == 0 {
		if len(catalogModels) > 0 {
			h.writeJSONCached(w, r, ck, 30, map[string]interface{}{"object": "list", "data": filterAllowed(catalogModels)})
			return
		}
		h.writeJSONCached(w, r, ck, 30, map[string]interface{}{"object": "list", "data": []interface{}{}})
		return
	}
	type modelResp struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			OwnedBy string `json:"owned_by"`
		} `json:"data"`
	}
	seen := map[string]bool{}
	var all []interface{}
	for _, cm := range catalogModels {
		id := cm["id"].(string)
		seen[id] = true
		all = append(all, cm)
	}
	for _, prov := range providers {
		apiKey, derr := h.ProviderStore.DecryptKey(&prov)
		if derr != nil {
			continue // skip providers whose credentials cannot be decrypted
		}
		target := strings.TrimRight(prov.BaseURL, "/") + "/models"
		if prov.Type == models.ProviderAnthropic {
			continue
		}
		req, rerr := http.NewRequestWithContext(r.Context(), "GET", target, nil)
		if rerr != nil {
			// Malformed operator-configured BaseURL: skip rather than
			// nil-deref on req below.
			log.Warn().Err(rerr).Str("provider", prov.Name).Str("target", target).Msg("models probe: bad upstream URL, skipping")
			continue
		}
		req.Header.Set("User-Agent", models.GatewayUserAgent)
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, err := h.Client.Do(req)
		if err != nil {
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
		resp.Body.Close()
		if resp.StatusCode != 200 {
			continue
		}
		var mr modelResp
		if json.Unmarshal(body, &mr) == nil {
			for _, m := range mr.Data {
				qid := qualifiedModelID(prov.Name, m.ID)
				if seen[qid] || seen[m.ID] {
					continue
				}
				seen[qid] = true
				seen[m.ID] = true
				owned := m.OwnedBy
				if owned == "" {
					owned = prov.Name
				}
				entry := map[string]interface{}{
					"id": qid, "model_id": shortModelID(m.ID), "object": m.Object, "owned_by": owned, "created": gatewayEpoch,
				}
				if h.CatalogStore != nil {
					if cm, err := h.CatalogStore.Get(m.ID); err == nil {
						entry["context_window"] = cm.ContextWindow
						entry["max_output"] = cm.MaxOutput
						entry["input_cost"] = cm.InputCost
						entry["output_cost"] = cm.OutputCost
						entry["reasoning"] = cm.Reasoning
					} else if cm, err := h.CatalogStore.GetByShortID(m.ID); err == nil {
						entry["context_window"] = cm.ContextWindow
						entry["max_output"] = cm.MaxOutput
						entry["input_cost"] = cm.InputCost
						entry["output_cost"] = cm.OutputCost
						entry["reasoning"] = cm.Reasoning
					}
				}
				all = append(all, entry)
			}
		}
	}
	if h.DB != nil {
		rows, _ := h.DB.Query(db.Q(`SELECT alias, target FROM model_aliases`))
		if rows != nil {
			defer rows.Close()
			for rows.Next() {
				var alias, target string
				if err := rows.Scan(&alias, &target); err != nil || alias == "" {
					continue
				}
				if !seen[alias] {
					seen[alias] = true
					all = append(all, map[string]interface{}{"id": alias, "object": "model", "owned_by": "gateway-alias", "alias_target": target, "created": gatewayEpoch})
				}
			}
		}
	}
	if len(all) == 0 {
		all = append(all, map[string]interface{}{"id": "gpt-4o", "object": "model", "owned_by": "gateway", "created": gatewayEpoch})
	}
	// apply allowlist to final aggregated list as well
	var filteredAll []interface{}
	if len(allowed) == 0 {
		filteredAll = all
	} else {
		for _, it := range all {
			var id string
			switch v := it.(type) {
			case map[string]interface{}:
				if s, ok := v["id"].(string); ok {
					id = s
				}
				// also allow if alias_target matches allowlist
				if id != "" && !apikey.IsModelAllowed(allowed, id) {
					if tgt, ok := v["alias_target"].(string); ok && apikey.IsModelAllowed(allowed, tgt) {
						id = tgt
					} else {
						continue
					}
				}
			case map[string]string:
				id = v["id"]
				if !apikey.IsModelAllowed(allowed, id) {
					continue
				}
			default:
				continue
			}
			if id != "" && !apikey.IsModelAllowed(allowed, id) {
				continue
			}
			filteredAll = append(filteredAll, it)
		}
		if len(filteredAll) == 0 && len(allowed) > 0 {
			// if nothing matches, return empty rather than leaking
			filteredAll = []interface{}{}
		}
	}
	h.writeJSONCached(w, r, ck, 30, map[string]interface{}{"object": "list", "data": filteredAll})
}

func (h *Handler) AnthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, bodyOK := h.readProxyBody(w, r)
	if !bodyOK {
		return
	}
	rawModel := translate.ExtractModel(body)
	model := h.resolveAlias(rawModel)
	if !h.enforceModelAllowlist(w, r, rawModel, model) {
		return
	}
	if !h.checkTPM(w, r, body) {
		return
	}
	if model != rawModel {
		body = replaceModelInBody(body, model)
	}
	// max_tokens must be integer-typed per the Anthropic spec: wrong types
	// crash strict upstreams and poison provider health. Type errors only —
	// value validation stays with the upstream.
	if !intTypeOK(body, "max_tokens") {
		httperr.Invalid(w, "invalid type for 'max_tokens': expected an integer")
		return
	}
	isStream := translate.IsStreaming(body)
	providerHint := r.Header.Get("X-Provider")
	candidates, rule := h.candidateProvidersWithRule(rawModel, model, providerHint, h.requestKeyOrg(r), func(p *models.Provider) bool {
		if p.Type == models.ProviderAnthropic {
			return true
		}
		if isAntigravity(p) {
			return true
		}
		if isDevin(p) {
			return true
		}
		// Multi-protocol providers (OpenCode Go/Zen) expose /v1/messages
		// alongside chat/responses on one base URL + key. Without this a
		// single entry cannot serve its messages-models (qwen/minimax).
		return isMultiProtocolProvider(p)
	})
	if len(candidates) == 0 {
		if h.unroutedModel() {
			noRouteFor(w, model)
			return
		}
		if p, err := h.ProviderStore.Resolve(model, providerHint); err == nil && p != nil && p.Type != models.ProviderAnthropic && !isMultiProtocolProvider(p) {
			httperr.Invalid(w, "model '"+model+"' is not an anthropic model; use POST /v1/chat/completions, /v1/completions or /v1/responses")
			return
		}
		// 503 is an availability signal, not a 404: proxy_error keeps SDK
		// retry classification and error vocabularies honest.
		httperr.Write(w, http.StatusServiceUnavailable, "no provider configured", httperr.TypeProxy)
		return
	}
	if err := h.validateReasoning(candidates[0].ID, model, body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	// Multi-protocol providers serve each model on exactly one upstream
	// endpoint. A messages-model called here is correct; a chat/responses
	// model is a client wiring error — fail fast with the right endpoint
	// instead of forwarding to /v1/messages for a cryptic upstream 400.
	if p0 := candidates[0].Provider; isMultiProtocolProvider(p0) {
		if api := UpstreamAPIForModel(p0, model); api == UpstreamChat || api == UpstreamResponses {
			httperr.Invalid(w, "model '"+model+"' is served via "+correctInboundFor(api)+" on this provider; use POST "+correctInboundFor(api)+" instead of /v1/messages")
			return
		}
	}
	// Antigravity via OAuth: normalize Anthropic -> OpenAI chat, then translate.
	if p0 := candidates[0].Provider; isAntigravity(p0) {
		chatBody, _, err := translate.AnthropicToOpenAI(body)
		if err != nil {
			httperr.Invalid(w, "invalid anthropic body: "+err.Error())
			return
		}
		start := time.Now()
		h.proxyAntigravity(w, r, body, chatBody, isStream, model, "messages", r.Header.Get("X-Gateway-Key-Prefix"), start, p0)
		return
	}
	// Devin via OAuth: normalize Anthropic -> OpenAI chat, then translate.
	if p0 := candidates[0].Provider; isDevin(p0) {
		chatBody, _, err := translate.AnthropicToOpenAI(body)
		if err != nil {
			httperr.Invalid(w, "invalid anthropic body: "+err.Error())
			return
		}
		start := time.Now()
		h.proxyDevin(w, r, chatBody, isStream, model, "messages", r.Header.Get("X-Gateway-Key-Prefix"), start, p0)
		return
	}
	start := time.Now()
	h.proxyCandidates(w, r, body, isStream, model, "messages", r.Header.Get("X-Gateway-Key-Prefix"), start, candidates, rule, func(p *models.Provider, body []byte) (string, string, []byte, bool, error) {
		apiKey, err := h.ProviderStore.DecryptKey(p)
		if err != nil {
			return "", "", nil, false, err
		}
		// The body may already carry a rule member's model override; derive
		// the upstream id from it so the override survives prefix stripping.
		bodyModel := translate.ExtractModel(body)
		upstream := stripProviderPrefix(bodyModel, p)
		out := body
		if upstream != bodyModel {
			out = replaceModelInBody(body, upstream)
		}
		target := strings.TrimRight(p.BaseURL, "/") + "/v1/messages"
		if strings.HasSuffix(p.BaseURL, "/v1/messages") {
			target = p.BaseURL
		} else if strings.Contains(p.BaseURL, "anthropic.com") && !strings.Contains(target, "/v1/messages") {
			target = strings.TrimRight(p.BaseURL, "/") + "/v1/messages"
		}
		return target, apiKey, out, true, nil
	})
}

// hasPreviousResponseID reports whether a /v1/responses body carries
// previous_response_id (conversation-continuation state the translated path
// cannot honor).
func hasPreviousResponseID(body []byte) bool {
	var probe struct {
		Prev *string `json:"previous_response_id"`
	}
	if json.Unmarshal(body, &probe) != nil {
		return false
	}
	return probe.Prev != nil && *probe.Prev != ""
}

// Responses handles POST /v1/responses (OpenAI Responses API)
func (h *Handler) Responses(w http.ResponseWriter, r *http.Request) {
	retry := h.retryOrDefault()
	body, bodyOK := h.readProxyBody(w, r)
	if !bodyOK {
		return
	}
	rawModel := translate.ExtractModel(body)
	model := h.resolveAlias(rawModel)
	if !h.enforceModelAllowlist(w, r, rawModel, model) {
		return
	}
	if !h.checkTPM(w, r, body) {
		return
	}
	if model != rawModel {
		body = replaceModelInBody(body, model)
	}
	// Spec-typed validation for the Responses params that translate into
	// chat max_tokens/n downstream: wrong types crash strict upstreams
	// (new-api "cannot unmarshal") and poison provider health.
	if err := validateResponsesParamTypes(body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	// stream_tool_calls is a client-side transport knob some agents put in
	// the body; the OpenAI Responses spec has no such parameter and strict
	// upstreams (OpenCode Go) 400 with "unknown parameter". It is never
	// meaningful to forward, so drop it before either upstream path.
	if stripped := deleteResponsesKey(body, "stream_tool_calls"); stripped != nil {
		body = stripped
	}
	isStream := translate.IsStreaming(body)
	providerHint := r.Header.Get("X-Provider")
	candidates, _ := h.candidateProvidersWithRule(rawModel, model, providerHint, h.requestKeyOrg(r), nil)
	if len(candidates) == 0 {
		if h.unroutedModel() {
			noRouteFor(w, model)
			return
		}
		httperr.Proxy(w, http.StatusServiceUnavailable, "no provider configured")
		return
	}
	p := candidates[0].Provider
	// Per-member model override from a curated rule (rule-routed traffic only).
	if override := candidates[0].ModelOverrideForRequest(); override != "" && override != model {
		body = replaceModelInBody(body, override)
		model = override
	}
	if err := h.validateReasoning(p.ID, model, body); err != nil {
		httperr.Invalid(w, err.Error())
		return
	}
	// Antigravity via OAuth: normalize Responses -> OpenAI chat, then translate.
	if isAntigravity(p) {
		chatBody, _, err := translate.ResponsesToChat(body)
		if err != nil {
			httperr.Invalid(w, "invalid responses body: "+err.Error())
			return
		}
		h.proxyAntigravity(w, r, body, chatBody, isStream, model, "responses", r.Header.Get("X-Gateway-Key-Prefix"), time.Now(), p)
		return
	}
	// Devin via OAuth: normalize Responses -> OpenAI chat, then translate.
	if isDevin(p) {
		chatBody, _, err := translate.ResponsesToChat(body)
		if err != nil {
			httperr.Invalid(w, "invalid responses body: "+err.Error())
			return
		}
		h.proxyDevin(w, r, chatBody, isStream, model, "responses", r.Header.Get("X-Gateway-Key-Prefix"), time.Now(), p)
		return
	}
	apiKey, derr := h.ProviderStore.DecryptKey(p)
	if derr != nil {
		httperr.Write(w, http.StatusBadGateway, "provider credential unavailable", httperr.TypeProxy)
		return
	}
	start := time.Now()
	keyPrefix := r.Header.Get("X-Gateway-Key-Prefix")
	// Same client-facing first-byte budget as the chat/messages paths: the
	// native probe and every translated attempt are bounded by the remaining
	// budget; streams that exhaust it get SSE keepalive frames, buffered
	// requests an honest 504 — all before a Cloudflare edge can 524.
	ttfb := newTTFBController(h.Timeouts.TTFB, start)
	defer ttfb.stop()
	w = &keepaliveSafeWriter{ResponseWriter: w, c: ttfb}
	// For anthropic providers, /v1/responses does not exist — skip native and translate directly.
	// Multi-protocol messages-models (OpenCode Go qwen/minimax) likewise
	// never speak /responses upstream; probing it wastes a call and logs noise.
	needsMessagesUpstream := p.Type == models.ProviderAnthropic || UpstreamAPIForModel(p, model) == UpstreamMessages
	if p.Type != models.ProviderAnthropic && !needsMessagesUpstream {
		isAzureUpstream := p.Type == models.ProviderAzure
		target := upstreamTarget(p.BaseURL, "/responses", model, isAzureUpstream)
		// Strip the "provider/model" routing prefix before probing native:
		// chat/completions/embeddings all do this; the native probe sent
		// "oc1/model" verbatim and strict upstreams 404'd model_not_found,
		// forcing a spurious fallback to /chat/completions (the 502s).
		upstreamModel := stripProviderPrefix(model, p)
		nativeBody := body
		if upstreamModel != translate.ExtractModel(body) {
			nativeBody = replaceModelInBody(body, upstreamModel)
		}
		req, rerr := http.NewRequestWithContext(r.Context(), "POST", target, bytes.NewReader(nativeBody))
		if rerr != nil {
			// Malformed operator-configured BaseURL: fall through to the
			// translated path instead of nil-dereferencing req below.
			log.Warn().Err(rerr).Str("provider", p.ID).Str("target", target).Msg("native responses probe: bad upstream URL")
		} else {
			if strings.Contains(target, "/openai/deployments/") {
				req.Header.Set("api-key", apiKey)
			} else {
				req.Header.Set("Authorization", "Bearer "+apiKey)
			}
			req.Header.Set("Content-Type", "application/json")
			// OpenCode Go/Zen requires the client's x-opencode-session for
			// routing/prompt-caching (400 MissingSessionID without it). The
			// native probe builds its request manually, so forward the
			// allowlisted session/user headers explicitly like
			// newUpstreamRequest does, synthesizing one when the client
			// sent none (opencodeSessionForRequest).
			for _, hk := range []string{"x-opencode-session", "user-agent", "accept-language", "x-request-id"} {
				if v := r.Header.Get(hk); v != "" {
					req.Header.Set(hk, v)
				}
			}
			// Same WAF guard as newUpstreamRequest: never send Go's default
			// User-Agent (or nothing) upstream — Cloudflare 403s it.
			if ua := req.Header.Get("User-Agent"); ua == "" || strings.HasPrefix(ua, "Go-http-client") {
				req.Header.Set("User-Agent", models.GatewayUserAgent)
			}
			if isOpencodeTarget(target) && req.Header.Get("x-opencode-session") == "" {
				req.Header.Set("x-opencode-session", opencodeSessionForRequest(r))
			}
			if isStream {
				// Ask for a real stream so SSE-native providers open one instead
				// of silently downgrading to a buffered JSON answer.
				req.Header.Set("Accept", "text/event-stream")
			} else {
				req.Header.Set("Accept", "application/json")
			}
			attemptCtx, attemptCancel := context.WithCancel(r.Context())
			defer attemptCancel()
			req = req.WithContext(attemptCtx)
			ttfbTimer := armTTFBWatchdog(ttfb, attemptCancel)
			resp, err := h.Client.Do(req)
			if ttfbTimer != nil {
				ttfbTimer.Stop()
			}
			// Transient probe failures (upstream 429/5xx) are retried within
			// the same budget before falling through: OpenCode Go rate-limits
			// muse-spark bursts with 429s, and a silent probe fall-through
			// lands on the translated /chat/completions path, which upstream
			// 500s for Responses-only models — turning a retryable upstream
			// hiccup into a hard client-visible failure.
			for probeAttempt := 0; err == nil && resp != nil && (resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500) && retry.ShouldRetry(probeAttempt, resp.StatusCode); probeAttempt++ {
				io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<20))
				resp.Body.Close()
				sleepCtx(r.Context(), retryAfterDelay(resp.Header, retry.Backoff(probeAttempt)))
				req2, rerr2 := http.NewRequestWithContext(r.Context(), "POST", target, bytes.NewReader(nativeBody))
				if rerr2 != nil {
					break
				}
				req2.Header = req.Header.Clone()
				req2 = req2.WithContext(attemptCtx)
				ttfbTimer := armTTFBWatchdog(ttfb, attemptCancel)
				resp, err = h.Client.Do(req2)
				if ttfbTimer != nil {
					ttfbTimer.Stop()
				}
			}
			// Only an explicit 200 is a usable native reply. Location-less 3xx
			// bodies arrive without transport error and previously passed through as
			// pseudo-success, bypassing translation. Redirects WITH Location are
			// followed by the http.Client before this point.
			if err == nil && resp.StatusCode == http.StatusOK {
				ct := resp.Header.Get("Content-Type")
				if isStream && strings.HasPrefix(ct, "text/event-stream") {
					// TRUE native pass-through streaming: relay chunk-by-chunk,
					// flush per read. A false return means nothing was committed
					// (dead-on-arrival upstream) — fall through to translated.
					if out := h.streamNativeResponses(w, r, resp, model, keyPrefix, p.ID, start, ttfb); out.committed {
						return
					}
					log.Info().Str("model", model).Str("provider", p.ID).Msg("native responses attempt unusable before commit; falling through to translated path")
				} else {
					bodyBytes, _ := io.ReadAll(resp.Body)
					resp.Body.Close()
					nativeSSE := resp.StatusCode == 200 &&
						(strings.HasPrefix(ct, "text/event-stream") || bytes.Contains(bodyBytes, []byte("\ndata: ")) || bytes.Contains(bodyBytes, []byte("data: ")))
					jsonOK := json.Valid(bodyBytes)
					noHTML := !bytes.Contains(bytes.ToLower(bodyBytes), []byte("<!doctype")) && !bytes.Contains(bytes.ToLower(bodyBytes), []byte("<html"))
					if !nativeSSE && jsonOK && noHTML && !hasResponsesOutput(bodyBytes) {
						log.Info().Str("model", model).Str("provider", p.ID).Msg("native responses JSON missing output; falling through to translated path")
					} else if (jsonOK && noHTML) || nativeSSE {
						copyHeader(w.Header(), resp.Header)
						w.WriteHeader(resp.StatusCode)
						w.Write(bodyBytes)
						pt, ct2 := extractUsage(bodyBytes)
						cost := h.costForModel(model, pt, ct2)
						h.logRequestExtended(keyPrefix, p.ID, model, "responses", resp.StatusCode, time.Since(start).Milliseconds(), pt, ct2, cost, false)
						// Billing parity: the native non-stream path previously
						// logged spend but never fed the budget ledger,
						// silently bypassing quotas.
						h.recordUsage(keyPrefix, r, pt+ct2, cost)
						return
					}
				}
			} else if resp != nil {
				// A 429 is endpoint-agnostic: the upstream rate-limits the
				// MODEL, so the translated path would hit the same limit —
				// and for Responses-only models (muse-spark-*) it is a
				// guaranteed 500 besides. Relay the 429 (and its Retry-After)
				// so client SDKs can back off, instead of masking it as a
				// translated-path 502.
				if resp.StatusCode == http.StatusTooManyRequests {
					bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
					resp.Body.Close()
					copyHeader(w.Header(), resp.Header)
					if w.Header().Get("Retry-After") == "" {
						w.Header().Set("Retry-After", "2")
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusTooManyRequests)
					if len(bodyBytes) > 0 {
						w.Write(bodyBytes)
					} else {
						w.Write([]byte(`{"error":{"code":"rate_limit_exceeded","type":"rate_limit_error","message":"upstream rate limit exceeded; retry shortly"}}`))
					}
					h.logRequestExtended(keyPrefix, p.ID, model, "responses", http.StatusTooManyRequests, time.Since(start).Milliseconds(), 0, 0, 0, false)
					return
				}
				// muse-spark-* models exist ONLY on the /responses endpoint;
				// the translated path upstream-500s for them, so a probe
				// rejection that falls through surfaces as an opaque 502.
				// Relay the upstream's real status/body so the client sees
				// the actual reason instead. Include the upstream error in
				// the log either way for diagnosability.
				responsesOnly := strings.HasPrefix(strings.ToLower(stripProviderPrefix(model, p)), "muse-spark-")
				errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
				log.Info().Str("model", model).Str("provider", p.ID).Int("status", resp.StatusCode).Str("err", errSnippet(errBody)).Msg("native responses probe rejected; falling through to translated path")
				resp.Body.Close()
				if responsesOnly {
					copyHeader(w.Header(), resp.Header)
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(resp.StatusCode)
					if len(errBody) > 0 {
						w.Write(errBody)
					} else {
						w.Write([]byte(fmt.Sprintf(`{"error":{"message":"upstream rejected this request with status %d","type":"proxy_error"}}`, resp.StatusCode)))
					}
					h.logRequestExtended(keyPrefix, p.ID, model, "responses", resp.StatusCode, time.Since(start).Milliseconds(), 0, 0, 0, false)
					return
				}
			}
		}
	}
	translated, _, err := translate.ResponsesToChat(body)
	if err != nil {
		httperr.Invalid(w, "failed to translate responses to chat: "+err.Error())
		return
	}
	// previous_response_id on the TRANSLATED path: native-capable upstreams
	// above already honored it themselves. Here the upstream never saw the
	// earlier turns, so expand the chain from the gateway's response store
	// (written by every translated turn) back into chat messages. An
	// unknown id — upstream-native turn, expired history, or a different
	// key's conversation — must refuse loudly: answering without the
	// conversation context hands clients confident, context-free guesses.
	prevID := extractPreviousResponseID(body)
	if prevID != "" {
		history, herr := h.loadResponseHistory(prevID, keyPrefix)
		if herr != nil {
			httperr.Invalid(w, "unknown or expired previous_response_id for this provider (no stored history); send the full conversation in input instead")
			return
		}
		var chatReq translate.OpenAIChatRequest
		if jerr := json.Unmarshal(translated, &chatReq); jerr != nil {
			httperr.Invalid(w, "failed to translate responses to chat: "+jerr.Error())
			return
		}
		chatReq.Messages = mergeResponseHistory(history, chatReq.Messages)
		merged, merr := json.Marshal(chatReq)
		if merr != nil {
			httperr.Invalid(w, "failed to translate responses to chat: "+merr.Error())
			return
		}
		translated = merged
	}
	// Shared Anthropic branch for BOTH stream and non-stream: the non-stream
	// path previously fell through, sending a chat-shaped body to
	// <anthropic-base>/chat/completions with Bearer auth — a guaranteed 404/401
	// against real Anthropic APIs.
	target := upstreamTarget(p.BaseURL, "/chat/completions", model, p.Type == models.ProviderAzure)
	upstreamBody := translated
	isAnthropicUpstream := false
	if p.Type == models.ProviderAnthropic || UpstreamAPIForModel(p, model) == UpstreamMessages {
		translated2, _, convErr := translate.OpenAIToAnthropic(translated)
		if convErr != nil || len(translated2) == 0 {
			httperr.Invalid(w, "failed to translate responses to anthropic")
			return
		}
		upstreamBody = translated2
		target = strings.TrimRight(p.BaseURL, "/") + "/v1/messages"
		isAnthropicUpstream = true
	} else {
		// The translated chat body goes to an OpenAI-compat upstream: apply
		// the same hygiene as the chat endpoint and strip the provider
		// prefix (chat does both; responses previously sent
		// "provider/model" verbatim, which strict/real upstreams reject).
		upstream := stripProviderPrefix(model, p)
		if upstream != translate.ExtractModel(upstreamBody) {
			upstreamBody = replaceModelInBody(upstreamBody, upstream)
		}
		upstreamBody = sanitizeOpenAICompatBodyOpts(upstreamBody, sanitizeOpts{keepAttachments: h.modelSupportsAttachment(p.ID, upstream)})
	}
	if isStream {
		// Translated-path streaming: force stream:true on the outbound chat
		// (or anthropic messages) call and re-emit inbound deltas as the
		// OpenAI Responses SSE protocol. LB guarantees a single candidate.
		// Chat upstreams need stream_options.include_usage or the streamed
		// tokens are invisible to budgets; Anthropic always emits usage
		// frames in message_start/message_delta.
		if !isAnthropicUpstream {
			if b2, changed := injectStreamUsage(upstreamBody); changed {
				upstreamBody = b2
			}
		}
		h.streamTranslatedResponses(w, r, target, apiKey, upstreamBody, model, keyPrefix, p.ID, start, isAnthropicUpstream, ttfb, prevID)
		return
	}
	out := h.proxyWithMetricsOpts(w, r, target, apiKey, upstreamBody, false, model, p.ID, keyPrefix, "responses", start, isAnthropicUpstream, proxyOpts{translatedResponses: true, ttfb: ttfb, prevResponseID: prevID})
	if !out.committed {
		// Client-caused upstream 5xx (invalid_request / convert_request_failed
		// semantics): relay the upstream's verdict instead of the generic
		// retryable-looking envelope.
		if out.clientCaused && out.errSnippet != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(outboundStatus(out.status))
			_, _ = w.Write([]byte(out.errSnippet))
			return
		}
		// Pre-commit terminal failure (e.g. upstream answered 200 with an
		// empty body): without this, nothing is written and net/http's
		// implicit 200 + empty body goes out — the exact bug that served
		// silent empty successes to /v1/responses clients.
		httperr.Proxy(w, outboundStatus(out.status), "upstream unavailable")
	}
}

// replaceModelInBody rewrites the top-level "model" field without a full
// map[string]interface{} round-trip. Re-encoding the entire body through
// interface{} coerces every JSON number to float64, silently corrupting
// int64 literals above 2^53 (seed is spec'd up to 2^63-1; large logit_bias
// values and metadata ids share the hazard) and normalizing key order. A
// targeted splice on the raw "model" value keeps every other byte intact.
func replaceModelInBody(body []byte, newModel string) []byte {
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return body
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return body // not an object: nothing to replace
	}
	var valStart, valEnd int64 = -1, -1
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return body
		}
		key, _ := keyTok.(string)
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return body
		}
		if key == "model" {
			var old string
			if json.Unmarshal(raw, &old) != nil {
				return body // non-string model: leave the body alone
			}
			// Byte offset of the raw value within the original body.
			valStart = dec.InputOffset() - int64(len(raw))
			valEnd = dec.InputOffset()
			break
		}
	}
	if valStart < 0 {
		return body
	}
	newRaw, err := json.Marshal(newModel)
	if err != nil {
		return body
	}
	out := make([]byte, 0, len(body)-int(valEnd-valStart)+len(newRaw))
	out = append(out, body[:valStart]...)
	out = append(out, newRaw...)
	out = append(out, body[valEnd:]...)
	return out
}
