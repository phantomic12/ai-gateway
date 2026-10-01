package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"ai-gateway/internal/apikey"
	"ai-gateway/internal/db"
	"ai-gateway/internal/lb"
	"ai-gateway/internal/middleware"
	"ai-gateway/internal/models"
	"ai-gateway/internal/provider"
	"ai-gateway/internal/resilience"

	"github.com/go-chi/chi/v5"
)

// lbHarness wires a gateway with two providers (a/b) both serving gpt-4o-mini,
// plus a third provider "pinme" used to prove pin-bypass and group exclusivity.
type lbHarness struct {
	srv         *httptest.Server
	h           *Handler
	key         string
	hitsA       atomic.Int32
	hitsB       atomic.Int32
	hitsPinme   atomic.Int32
	bodyA       string // returns 200 "from-a"
	bodyB       string // returns 200 "from-b"
	failA       bool
	failAStatus int // status used when failA (default 500)
	lbStore     *lb.Store
	paID        string
	pbID        string
	ppID        string
}

func newLBHarness(t *testing.T, ruleProviders []string, failA bool) *lbHarness {
	t.Helper()
	database, err := db.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	ps := provider.NewStore(database, make([]byte, 32))
	ks := apikey.NewStore(database)

	hh := &lbHarness{failA: failA, bodyA: "from-a", bodyB: "from-b"}
	upA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hh.hitsA.Add(1)
		if hh.failA {
			status := hh.failAStatus
			if status == 0 {
				status = 500
			}
			w.WriteHeader(status)
			w.Write([]byte(`{"error":{"message":"a down"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "cmpl-a", "choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": hh.bodyA}}},
			"usage": map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	upB := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hh.hitsB.Add(1)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "cmpl-b", "choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": hh.bodyB}}},
			"usage": map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	upPin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hh.hitsPinme.Add(1)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "cmpl-p", "choices": []map[string]interface{}{{"message": map[string]string{"role": "assistant", "content": "from-pinme"}}},
			"usage": map[string]int{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	}))
	t.Cleanup(func() { upA.Close(); upB.Close(); upPin.Close() })

	pa, err := ps.Create("prov-a", models.ProviderOpenAI, upA.URL+"/v1", "sk-a")
	if err != nil {
		t.Fatal(err)
	}
	pb, err := ps.Create("prov-b", models.ProviderOpenAI, upB.URL+"/v1", "sk-b")
	if err != nil {
		t.Fatal(err)
	}
	pp, err := ps.Create("pinme", models.ProviderOpenAI, upPin.URL+"/v1", "sk-p")
	if err != nil {
		t.Fatal(err)
	}
	now := "2026-01-01T00:00:00Z"
	for _, id := range []string{pa.ID, pb.ID} {
		if _, err := database.Exec(`INSERT INTO provider_models(id, provider_id, model_id, display_name, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?)`, id+"-m", id, "gpt-4o-mini", "gpt-4o-mini", "manual", now, now); err != nil {
			t.Fatal(err)
		}
	}

	k, err := ks.Create("test-key")
	if err != nil {
		t.Fatal(err)
	}
	hh.key = k.Key

	h := New(ps, database)
	h.Retry = &resilience.DefaultRetryPolicy{MaxRetries: 0}
	h.LB = lb.NewStore(database)
	hh.h = h
	hh.lbStore = h.LB
	hh.paID, hh.pbID, hh.ppID = pa.ID, pb.ID, pp.ID

	r := chi.NewRouter()
	r.Use(middleware.GatewayAuth(ks))
	r.Post("/v1/chat/completions", h.ChatCompletions)
	hh.srv = httptest.NewServer(r)
	t.Cleanup(hh.srv.Close)

	if ruleProviders != nil {
		// Translate provider names → ids, mirroring what the dashboard does.
		nameToID := map[string]string{pa.Name: pa.ID, pb.Name: pb.ID, pp.Name: pp.ID}
		members := make([]lb.RuleMemberInput, 0, len(ruleProviders))
		for _, n := range ruleProviders {
			id, ok := nameToID[n]
			if !ok {
				t.Fatalf("unknown test provider %q", n)
			}
			members = append(members, lb.RuleMemberInput{ProviderID: id})
		}
		if err := h.LB.ReplaceRule("gpt-4o-mini", "", members); err != nil {
			t.Fatal(err)
		}
	}
	return hh
}

// do sends a chat completion with optional X-Provider pin and qualified model.
func (hh *lbHarness) do(t *testing.T, model, xProvider string) (int, string) {
	t.Helper()
	req, _ := http.NewRequest("POST", hh.srv.URL+"/v1/chat/completions", strings.NewReader(`{"model":"`+model+`","messages":[{"role":"user","content":"Hello"}]}`))
	req.Header.Set("Authorization", "Bearer "+hh.key)
	req.Header.Set("Content-Type", "application/json")
	if xProvider != "" {
		req.Header.Set("X-Provider", xProvider)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	buf := make([]byte, 8192)
	n, _ := resp.Body.Read(buf)
	return resp.StatusCode, string(buf[:n])
}

// Regression: cross-provider failover is DISABLED by product decision. When
// the selected provider fails, the error surfaces after same-provider retry
// policy exhausts — the healthy sibling is NEVER contacted.
func TestNoCrossProviderFailover(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	// Pin prov-a explicitly so selection is deterministic.
	hh.failA = true

	code, body := hh.do(t, "gpt-4o-mini", "prov-a")
	if code < 500 || strings.Contains(body, "from-b") {
		t.Fatalf("expected honest failure from prov-a, got %d %s", code, body)
	}
	if n := hh.hitsB.Load(); n != 0 {
		t.Fatalf("fallback is banned: prov-b was contacted %d times", n)
	}
	if hh.hitsA.Load() == 0 {
		t.Fatal("primary should have been attempted")
	}
}

// Round-robin rotates ACROSS requests through curated group members only;
// providers serving the same model outside the rule are never touched, and a
// failing member yields its own error without invoking siblings.
func TestLBRoundRobinAcrossRequests(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)

	var seenA, seenB bool
	for i := 0; i < 6; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 {
			t.Fatalf("request %d failed: %d %s", i, code, body)
		}
		if strings.Contains(body, "from-a") {
			seenA = true
		}
		if strings.Contains(body, "from-b") {
			seenB = true
		}
	}
	if !seenA || !seenB {
		t.Fatalf("rotation must visit every member (seenA=%v seenB=%v)", seenA, seenB)
	}
	a, b := hh.hitsA.Load(), hh.hitsB.Load()
	if a == 0 || b == 0 || a+b != 6 {
		t.Fatalf("expected strict 3/3 split across 6 requests, got A=%d B=%d", a, b)
	}
}

// Curated groups are exclusive: another provider serving the identical bare
// model never receives traffic once a rule exists.
func TestLBGroupIsExclusive(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)
	for i := 0; i < 6; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 {
			t.Fatalf("req %d: %d %s", i, code, body)
		}
	}
	if n := hh.hitsPinme.Load(); n != 0 {
		t.Fatalf("out-of-group provider served %d requests", n)
	}
}

// Pins bypass groups entirely: qualified IDs route to the named provider even
// when a rule governs the base model.
func TestQualifiedPinBeatsRule(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)
	code, body := hh.do(t, "pinme/gpt-4o-mini", "")
	if code != 200 || !strings.Contains(body, "from-pinme") {
		t.Fatalf("qualified pin should route to pinme: %d %s", code, body)
	}
	if hh.hitsPinme.Load() != 1 || hh.hitsA.Load()+hh.hitsB.Load() != 0 {
		t.Fatalf("rule members must be untouched on pinned requests (A=%d B=%d)", hh.hitsA.Load(), hh.hitsB.Load())
	}
}

// Mixed-case display names ("AIHubMix") must pin from qualified IDs and
// X-Provider hints in any casing — the qualified prefix is lowercased before
// lookup while the stored name keeps its display case.
func TestQualifiedPinCaseInsensitive(t *testing.T) {
	database, err := db.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	ps := provider.NewStore(database, make([]byte, 32))
	pa, err := ps.Create("MixedCase", models.ProviderOpenAI, "https://example.com/v1", "sk-x")
	if err != nil {
		t.Fatal(err)
	}
	h := New(ps, database)
	h.LB = lb.NewStore(database)

	for _, model := range []string{"MixedCase/m", "mixedcase/m", "MIXEDCASE/m"} {
		cands, _ := h.candidateProvidersWithRule(model, model, "", "", nil)
		if len(cands) != 1 || cands[0].ID != pa.ID {
			t.Fatalf("qualified pin %q missed MixedCase (got %d candidates)", model, len(cands))
		}
	}
	for _, hint := range []string{"MixedCase", "mixedcase", "MIXEDCASE"} {
		cands, _ := h.candidateProvidersWithRule("m", "m", hint, "", nil)
		if len(cands) != 1 || cands[0].ID != pa.ID {
			t.Fatalf("X-Provider hint %q missed MixedCase (got %d candidates)", hint, len(cands))
		}
	}
}

// Failing member surfaces its own error; rotation continues on LATER requests
// (operators remove dead members from the UI rather than relying on silent
// failover).
func TestLBFailingMemberErrorsHonest(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, true)
	var gotErr, gotOK int
	for i := 0; i < 6; i++ {
		code, _ := hh.do(t, "gpt-4o-mini", "")
		if code >= 500 {
			gotErr++
		} else if code == 200 {
			gotOK++
		}
	}
	if gotErr != 3 || gotOK != 3 {
		t.Fatalf("strict per-request rotation expected 3 fails / 3 oks, got %d/%d", gotErr, gotOK)
	}
}

// Bare model names nobody owns (no routing rule, no provider_models
// ownership, no pin) are rejected with 404 model_not_routed (strict default)
// — no upstream is contacted.
func TestBareModelWithoutRuleRejected(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	code, body := hh.do(t, "unrouted-model", "")
	if code != 404 {
		t.Fatalf("expected 404 for unrouted bare model, got %d: %s", code, body)
	}
	if !strings.Contains(body, "model_not_routed") {
		t.Fatalf("expected model_not_routed code in body: %s", body)
	}
	if hh.hitsA.Load()+hh.hitsB.Load()+hh.hitsPinme.Load() != 0 {
		t.Fatal("unrouted model must not touch any upstream")
	}
}

// A model advertised by several providers but absent from lb_rules is NOT an
// error: the owner set serves it, rotated per request so load spreads, and
// a failing owner fails over to the next. This is the identical-slug case
// that used to return model_not_routed forever.
func TestIdenticalSlugAcrossProvidersServes(t *testing.T) {
	hh := newLBHarness(t, nil, false) // both prov-a and prov-b own gpt-4o-mini
	var a, b int
	for i := 0; i < 6; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 {
			t.Fatalf("req %d: identical-slug model must serve, got %d: %s", i, code, body)
		}
		if strings.Contains(body, "from-a") {
			a++
		} else if strings.Contains(body, "from-b") {
			b++
		}
	}
	if a == 0 || b == 0 {
		t.Fatalf("rotation expected both owners to serve (a=%d b=%d)", a, b)
	}
	if hh.hitsPinme.Load() != 0 {
		t.Fatal("non-member provider must not be touched")
	}
}

// Identical-slug failover: the rotated first owner 403s (reseller "access
// disabled" — retriable) → the request still succeeds from the other owner.
func TestIdenticalSlugFailsOverAcrossOwners(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	hh.failA = true // prov-a 500s every request
	hh.failAStatus = 403
	// Rotation may start at either owner; every request must still land on b.
	for i := 0; i < 4; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 || !strings.Contains(body, "from-b") {
			t.Fatalf("req %d: dead owner must fail over to the live one, got %d: %s", i, code, body)
		}
	}
}

// No provider owns the model: strict mode must not silently fall through to
// the default provider.
func TestUnownedModelDoesNotUseDefaultProvider(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	code, body := hh.do(t, "totally-unowned-model", "")
	if code != 404 || !strings.Contains(body, "model_not_routed") {
		t.Fatalf("unowned models must be rejected, not defaulted: %d %s", code, body)
	}
}

// ROUTING_LEGACY_FALLBACK restores the pre-strategy heuristic resolution.
func TestLegacyFallbackFlagRestoresHeuristics(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	hh.h.LegacyFallback = true
	code, body := hh.do(t, "gpt-4o-mini", "")
	if code != 200 {
		t.Fatalf("legacy fallback should resolve via provider_models ownership, got %d: %s", code, body)
	}
	if !strings.Contains(body, "from-") {
		t.Fatalf("expected a member response: %s", body)
	}
}

// failover strategy: primary down → next member serves, X-Fallback-Used set.
func TestFailoverStrategyFailsOver(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)
	hh.failA = true
	if err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyFailover, []lb.RuleMemberInput{
		{ProviderID: hh.paID},
		{ProviderID: hh.pbID},
	}); err != nil {
		t.Fatal(err)
	}
	code, body := hh.do(t, "gpt-4o-mini", "")
	if code != 200 || !strings.Contains(body, "from-b") {
		t.Fatalf("failover rule should serve from prov-b, got %d: %s", code, body)
	}
	if n := hh.hitsA.Load(); n == 0 {
		t.Fatal("primary should have been attempted first")
	}
}

// failover strategy: healthy primary → later members never contacted.
func TestFailoverStrategyNoFailoverWhenHealthy(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)
	if err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyFailover, []lb.RuleMemberInput{
		{ProviderID: hh.paID},
		{ProviderID: hh.pbID},
	}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 || !strings.Contains(body, "from-a") {
			t.Fatalf("healthy primary must keep serving: req %d got %d %s", i, code, body)
		}
	}
	if n := hh.hitsB.Load(); n != 0 {
		t.Fatalf("failover must not touch secondaries while primary is healthy: %d", n)
	}
}

// weighted strategy distributes the primary pick proportionally to weight.
func TestWeightedRuleDistribution(t *testing.T) {
	hh := newLBHarness(t, []string{"prov-a", "prov-b"}, false)
	if err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyWeighted, []lb.RuleMemberInput{
		{ProviderID: hh.paID, Weight: 90},
		{ProviderID: hh.pbID, Weight: 10},
	}); err != nil {
		t.Fatal(err)
	}
	const n = 200
	for i := 0; i < n; i++ {
		if code, _ := hh.do(t, "gpt-4o-mini", ""); code != 200 {
			t.Fatalf("req %d failed", i)
		}
	}
	a, b := hh.hitsA.Load(), hh.hitsB.Load()
	// 90/10 expectation with generous binomial bounds.
	if a < 150 || a > n {
		t.Fatalf("weighted 90/10: prov-a got %d/%d", a, n)
	}
	if b < 1 || b > 50 {
		t.Fatalf("weighted 90/10: prov-b got %d/%d", b, n)
	}
}

// model_override rewrites the outbound model for that member's upstream.
func TestRuleMemberModelOverride(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	// prov-a serves the rule; override asks upstream for a different model id.
	if err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyFailover, []lb.RuleMemberInput{
		{ProviderID: hh.paID, ModelOverride: "gpt-4o-mini-2024-07-18"},
	}); err != nil {
		t.Fatal(err)
	}

	var seenModel string
	hh.h.Client.Transport = captureModelTransport{fn: func(m string) { seenModel = m }}

	code, body := hh.do(t, "gpt-4o-mini", "")
	if code != 200 || !strings.Contains(body, "from-a") {
		t.Fatalf("override rule should serve: %d %s", code, body)
	}
	if seenModel != "gpt-4o-mini-2024-07-18" {
		t.Fatalf("upstream should receive overridden model, got %q", seenModel)
	}
}

// captureModelTransport observes the outbound "model" field of the JSON body.
type captureModelTransport struct {
	fn func(model string)
}

func (c captureModelTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.Body != nil {
		b, _ := io.ReadAll(req.Body)
		var m struct {
			Model string `json:"model"`
		}
		if json.Unmarshal(b, &m) == nil {
			c.fn(m.Model)
		}
		req.Body = io.NopCloser(bytes.NewReader(b))
	}
	return http.DefaultTransport.RoundTrip(req)
}

// A rule may list the SAME provider several times with different model
// overrides — one member per (provider, model) option. Round-robin must
// alternate the options and send each member's own model upstream (the
// provider id alone cannot decide the model).
func TestRuleMultipleModelsSameProvider(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	if err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyRoundRobin, []lb.RuleMemberInput{
		{ProviderID: hh.paID, ModelOverride: "gpt-4o-mini-2024-07-18"},
		{ProviderID: hh.paID, ModelOverride: "gpt-4o-mini-2024-10-22"},
		{ProviderID: hh.pbID, ModelOverride: "llama-b"},
	}); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	seen := map[string]int{}
	hh.h.Client.Transport = captureModelTransport{fn: func(m string) {
		mu.Lock()
		seen[m]++
		mu.Unlock()
	}}

	for i := 0; i < 9; i++ {
		code, body := hh.do(t, "gpt-4o-mini", "")
		if code != 200 {
			t.Fatalf("request %d failed: %d %s", i, code, body)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	for _, want := range []string{"gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-10-22", "llama-b"} {
		if seen[want] == 0 {
			t.Fatalf("member model %q never served (seen %v)", want, seen)
		}
	}
	if seen["gpt-4o-mini"] != 0 {
		t.Fatalf("requested model must be overridden on every member (seen %v)", seen)
	}
	if total := seen["gpt-4o-mini-2024-07-18"] + seen["gpt-4o-mini-2024-10-22"] + seen["llama-b"]; total != 9 {
		t.Fatalf("expected 9 upstream calls, counted %d (%v)", total, seen)
	}
}

// Exact duplicate (provider, model) members are rejected at write time.
func TestRuleRejectsDuplicateMember(t *testing.T) {
	hh := newLBHarness(t, nil, false)
	err := hh.lbStore.ReplaceRule("gpt-4o-mini", lb.StrategyRoundRobin, []lb.RuleMemberInput{
		{ProviderID: hh.paID, ModelOverride: "same-model"},
		{ProviderID: hh.paID, ModelOverride: "same-model"},
	})
	if err == nil {
		t.Fatal("exact duplicate member must be rejected")
	}
}
