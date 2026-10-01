package discovery

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"ai-gateway/internal/cache"
	"ai-gateway/internal/catalog"
	"ai-gateway/internal/db"
	"ai-gateway/internal/models"
	"ai-gateway/internal/provider"

	"github.com/google/uuid"
)

type Service struct {
	db            *sql.DB
	providerStore *provider.Store
	catalogStore  *catalog.Store
	client        *http.Client
	Cache         cache.Cache
}

func New(db *sql.DB, ps *provider.Store, cs *catalog.Store) *Service {
	return &Service{
		db: db, providerStore: ps, catalogStore: cs,
		client: &http.Client{Timeout: 15 * time.Second},
	}
}

type rawModelList struct {
	Data []struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
		Created int64  `json:"created"`
	} `json:"data"`
	Object string `json:"object"`
}

// parseModelList decodes a /models response body of either {data:[...]}
// shape (OpenAI list or Anthropic list) into rawModels, capturing both the
// identity fields and any provider-DECLARED detail (pricing, limits,
// capabilities). Providers that only list ids get nil Meta — the models.dev
// catalog stays the fallback source for those, never an override of what the
// provider itself says.
func parseModelList(body []byte) []rawModel {
	var list struct {
		Data []map[string]interface{} `json:"data"`
	}
	if json.Unmarshal(body, &list) != nil {
		return nil
	}
	var out []rawModel
	for _, mm := range list.Data {
		id, _ := mm["id"].(string)
		if id == "" {
			// Anthropic-style entries without id (rare): display_name is the
			// only usable identifier.
			id, _ = mm["display_name"].(string)
		}
		if id == "" {
			continue
		}
		rm := rawModel{ID: id}
		if dn, _ := mm["display_name"].(string); dn != "" && dn != id {
			rm.DisplayName = dn
		}
		if owned, _ := mm["owned_by"].(string); owned != "" {
			rm.OwnedBy = owned
		}
		rm.Meta = providerMetaFrom(mm)
		out = append(out, rm)
	}
	return out
}

// rawModel is one entry of a provider's model listing. Meta is non-nil only
// when the entry itself declared detail (OpenRouter pricing/limits,
// Anthropic display names, OpenAI-compatible capability flags).
type rawModel struct {
	ID          string
	OwnedBy     string
	DisplayName string
	Meta        *providerMeta
}

// providerMeta is provider-declared model detail harvested from the
// /v1/models listing object. Pointer/zero semantics separate "declared as
// zero" (a free route legitimately prices at 0) from "not declared": costs
// use *float64, sizes use >0.
type providerMeta struct {
	ContextWindow, MaxOutput int
	InputCost                *float64 // per 1M tokens
	OutputCost               *float64
	CacheReadCost            *float64
	CacheWriteCost           *float64
	Reasoning                *bool
	ToolCall                 *bool
	StructuredOutput         *bool
	Attachment               *bool
	Modalities               string // JSON array string, e.g. ["text","image"]
}

func strNum(v interface{}) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
		if err != nil {
			return 0, false
		}
		return f, true
	}
	return 0, false
}

func asMap(v interface{}) map[string]interface{} {
	m, _ := v.(map[string]interface{})
	return m
}

// providerMetaFrom extracts declared detail from one listing entry. Keys
// cover the shapes seen in the wild: flat generic fields
// (context_length/max_output_tokens/tool_call/...), OpenRouter's split
// (pricing.* per token, architecture.* modalities, top_provider.* limits,
// supported_parameters capability list) and LiteLLM-style booleans. Costs
// are normalized to per-1M to match provider_models semantics.
func providerMetaFrom(mm map[string]interface{}) *providerMeta {
	m := &providerMeta{}
	declared := false
	for _, keys := range [2][]string{
		{"context_length", "context_window", "max_context_length", "max_model_len"},
		{"max_output_tokens", "max_completion_tokens", "max_tokens"},
	} {
		for _, k := range keys {
			if v, ok := strNum(mm[k]); ok && v > 0 {
				if keys[0] == "context_length" {
					if m.ContextWindow == 0 {
						m.ContextWindow = int(v)
						declared = true
					}
				} else if m.MaxOutput == 0 {
					m.MaxOutput = int(v)
					declared = true
				}
			}
		}
	}
	if tp := asMap(mm["top_provider"]); tp != nil {
		if v, ok := strNum(tp["context_length"]); ok && v > 0 && m.ContextWindow == 0 {
			m.ContextWindow = int(v)
			declared = true
		}
		if v, ok := strNum(tp["max_completion_tokens"]); ok && v > 0 && m.MaxOutput == 0 {
			m.MaxOutput = int(v)
			declared = true
		}
	}
	// Flat costs are already per-1M; pricing.* is per-token and scaled ×1e6.
	num := func(dst **float64, k string) {
		if v, ok := strNum(mm[k]); ok && v >= 0 {
			*dst = &v
			declared = true
		}
	}
	num(&m.InputCost, "input_cost")
	num(&m.OutputCost, "output_cost")
	num(&m.CacheReadCost, "cache_read_cost")
	num(&m.CacheWriteCost, "cache_write_cost")
	if pr := asMap(mm["pricing"]); pr != nil {
		perM := func(dst **float64, k string) {
			if v, ok := strNum(pr[k]); ok && v >= 0 {
				// Per-token → per-1M. ×1e6 produces float noise
				// (0.0000002*1e6 = 0.19999999999999998); round to
				// enough digits for any real pricing ($0.000001/1M).
				per := math.Round(v*1_000_000*1e9) / 1e9
				*dst = &per
				declared = true
			}
		}
		if m.InputCost == nil {
			perM(&m.InputCost, "prompt")
		}
		if m.OutputCost == nil {
			perM(&m.OutputCost, "completion")
		}
		if m.CacheReadCost == nil {
			perM(&m.CacheReadCost, "input_cache_read")
		}
		if m.CacheWriteCost == nil {
			perM(&m.CacheWriteCost, "input_cache_write")
		}
	}
	boolFrom := func(dst **bool, k string) {
		if v, ok := mm[k].(bool); ok {
			*dst = &v
			declared = true
		}
	}
	boolFrom(&m.Reasoning, "reasoning")
	boolFrom(&m.ToolCall, "tool_call")
	boolFrom(&m.StructuredOutput, "structured_output")
	boolFrom(&m.Attachment, "attachment")
	// LiteLLM-style capability keys.
	boolFrom(&m.ToolCall, "supports_function_calling")
	boolFrom(&m.ToolCall, "supports_tool_choice")
	boolFrom(&m.StructuredOutput, "supports_response_schema")
	boolFrom(&m.Attachment, "supports_vision")
	// OpenRouter-style capability list: ["tools","structured_outputs",
	// "image_inputs","reasoning",...] — only fills what's absent, never
	// contradicts an explicit boolean above.
	if params, ok := mm["supported_parameters"].([]interface{}); ok {
		has := func(want string) bool {
			for _, p := range params {
				if s, _ := p.(string); s == want {
					return true
				}
			}
			return false
		}
		t := true
		if m.ToolCall == nil && has("tools") {
			m.ToolCall = &t
			declared = true
		}
		if m.StructuredOutput == nil && (has("structured_outputs") || has("response_format")) {
			m.StructuredOutput = &t
			declared = true
		}
		if m.Reasoning == nil && has("reasoning") {
			m.Reasoning = &t
			declared = true
		}
		if m.Attachment == nil && has("image_inputs") {
			m.Attachment = &t
			declared = true
		}
	}
	if arch := asMap(mm["architecture"]); arch != nil {
		// input_modalities is authoritative: non-text inputs → attachment.
		if mods, ok := arch["input_modalities"].([]interface{}); ok && len(mods) > 0 {
			nonText := false
			for _, m2 := range mods {
				if s2, _ := m2.(string); s2 != "" && s2 != "text" {
					nonText = true
				}
			}
			if m.Attachment == nil && nonText {
				t := true
				m.Attachment = &t
				declared = true
			}
			if b, err := json.Marshal(mods); err == nil {
				m.Modalities = string(b)
				declared = true
			}
		} else if mod, _ := arch["modality"].(string); mod != "" {
			// OpenRouter writes "text->text"; some write "text+image".
			parts := strings.FieldsFunc(mod, func(r rune) bool { return r == '+' || r == '-' || r == '>' })
			nonText := false
			for _, s := range parts {
				if s != "text" {
					nonText = true
				}
			}
			if m.Attachment == nil && nonText {
				t := true
				m.Attachment = &t
				declared = true
			}
			if b, err := json.Marshal(parts); err == nil {
				m.Modalities = string(b)
				declared = true
			}
		}
	}
	if !declared {
		return nil
	}
	return m
}

// Discover fetches /v1/models from provider and upserts provider_models, enriching from catalog
func (s *Service) Discover(providerID string) (int, error) {
	p, err := s.providerStore.GetByID(providerID)
	if err != nil {
		return 0, err
	}
	// Antigravity uses OAuth + Cloud Code Assist catalog, not /v1/models.
	if p.Type == models.ProviderAntigravity {
		return s.discoverAntigravity(p)
	}
	// Devin uses OAuth + Connect-proto model configs, not /v1/models.
	if p.Type == models.ProviderDevin {
		return s.discoverDevin(p)
	}
	apiKey, err := s.providerStore.DecryptKey(p)
	if err != nil {
		return 0, err
	}
	var fetched []rawModel
	// Multi-protocol providers (OpenCode Go/Zen) list different models per
	// endpoint family. Probe both dialects and merge so one provider entry
	// discovers its chat, responses, and messages models together.
	if isMultiProvider(p) {
		seen := map[string]bool{}
		for _, m := range s.fetchOpenAI(p, apiKey) {
			if !seen[m.ID] {
				seen[m.ID] = true
				fetched = append(fetched, m)
			}
		}
		for _, m := range s.fetchAnthropic(p, apiKey) {
			if !seen[m.ID] {
				seen[m.ID] = true
				fetched = append(fetched, m)
			}
		}
	} else {
		switch p.Type {
		case models.ProviderAnthropic:
			fetched = s.fetchAnthropic(p, apiKey)
		case models.ProviderAzure:
			fetched = s.fetchAzure(p, apiKey)
			if len(fetched) == 0 {
				fetched = s.fetchOpenAI(p, apiKey)
			}
		default:
			fetched = s.fetchOpenAI(p, apiKey)
			if len(fetched) == 0 && p.Type == models.ProviderAnthropic {
				fetched = s.fetchAnthropic(p, apiKey)
			}
		}
	}
	if len(fetched) == 0 {
		return 0, fmt.Errorf("no models discovered (check provider base_url and key)")
	}
	count := 0
	probeBudget := 8 // cap context probes per provider per run — each costs a request
	for _, m := range fetched {
		// Providers whose /models carries no detail (kourier et al.) get
		// catalog numbers describing the model's FULL capacity, not what
		// this reseller serves (kourier caps deepseek-v4.1-flash at 262k,
		// catalog says 1M). Probe once per bare-listed model until budget
		// is spent; providers that declared their own numbers are trusted.
		if err := s.upsert(p, m, apiKey, &probeBudget); err == nil {
			count++
		}
	}
	if s.Cache != nil && count > 0 {
		s.Cache.Invalidate("models:")
	}
	return count, nil
}

func (s *Service) fetchOpenAI(p *models.Provider, apiKey string) []rawModel {
	target := strings.TrimRight(p.BaseURL, "/") + "/models"
	// if base_url ends with /v1/models already? our store normalizes base_url without trailing slash, e.g., https://ckff.dev/v1, then +/models = /v1/models correct. If base_url is https://ckff.dev, +/models = /models wrong. So try both.
	urls := []string{target}
	if !strings.Contains(p.BaseURL, "/v1") {
		urls = append(urls, strings.TrimRight(p.BaseURL, "/")+"/v1/models")
	}
	for _, u := range urls {
		req, _ := http.NewRequest("GET", u, nil)
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, err := s.client.Do(req)
		if err != nil || resp.StatusCode != 200 {
			if resp != nil {
				resp.Body.Close()
			}
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 5<<20))
		resp.Body.Close()
		if out := parseModelList(body); len(out) > 0 {
			return out
		}
	}
	return nil
}

func (s *Service) fetchAzure(p *models.Provider, apiKey string) []rawModel {
	base := strings.TrimRight(p.BaseURL, "/")
	urls := []string{base + "/models?api-version=2024-02-01", base + "/models"}
	for _, u := range urls {
		req, _ := http.NewRequest("GET", u, nil)
		req.Header.Set("api-key", apiKey)
		resp, err := s.client.Do(req)
		if err != nil || resp.StatusCode != 200 {
			if resp != nil {
				resp.Body.Close()
			}
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 5<<20))
		resp.Body.Close()
		if out := parseModelList(body); len(out) > 0 {
			return out
		}
	}
	return nil
}

func (s *Service) fetchAnthropic(p *models.Provider, apiKey string) []rawModel {
	base := strings.TrimRight(p.BaseURL, "/")
	var urls []string
	switch {
	case strings.HasSuffix(base, "/v1/models"):
		urls = []string{base}
	case strings.Contains(base, "/v1"):
		// Base already carries a version prefix (e.g. https://ckff.dev/v1):
		// appending another /v1 would build /v1/v1/models (404).
		urls = []string{base + "/models"}
	default:
		urls = []string{base + "/v1/models"}
	}
	for _, target := range urls {
		req, _ := http.NewRequest("GET", target, nil)
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
		resp, err := s.client.Do(req)
		if err != nil || resp.StatusCode != 200 {
			if resp != nil {
				resp.Body.Close()
			}
			continue
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 5<<20))
		resp.Body.Close()
		if out := parseModelList(body); len(out) > 0 {
			return out
		}
	}
	return nil
}

func (s *Service) upsert(p *models.Provider, m rawModel, apiKey string, probeBudget *int) error {
	e := s.resolveEnrichment(m)
	// Providers whose /models carries no detail (kourier et al.) get the
	// model's catalog numbers — which describe the model's FULL capacity,
	// not what this reseller actually serves (kourier caps deepseek-v4.1-
	// flash at 262k, catalog says 1M). Probe once per bare-listed model
	// while budget lasts; providers that declared numbers are trusted.
	if m.Meta == nil && probeBudget != nil && *probeBudget > 0 {
		*probeBudget--
		if probe := s.probeContextLimit(p, apiKey, m.ID); probe > 0 {
			if e.ctx == 0 || probe < e.ctx {
				e.ctx = probe
			}
			if e.maxOut == 0 || probe < e.maxOut {
				e.maxOut = probe
			}
			e.source = "provider"
		}
	}
	// check existing to preserve manual overrides
	var existingID string
	var existingSource string
	err := s.db.QueryRow(db.Q(`SELECT id, source FROM provider_models WHERE provider_id=? AND model_id=?`), p.ID, m.ID).Scan(&existingID, &existingSource)
	if err == nil && existingSource == "manual" {
		// don't overwrite manual
		return nil
	}
	if err == sql.ErrNoRows && s.isExcluded(p.ID, m.ID) {
		// Operator removed this model; it comes back only from the recycling
		// bin or a manual add, never from discovery.
		return nil
	}
	displayName := m.ID
	if m.DisplayName != "" {
		displayName = m.DisplayName
	}
	if err == nil {
		_, err = s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, structured_output=?, attachment=?, modalities=?, reasoning_type=?, reasoning_levels=?, reasoning_output_limits=?, source=?, updated_at=? WHERE id=?`),
			displayName, m.OwnedBy, e.ctx, e.maxOut, e.inputCost, e.outputCost, e.cacheReadCost, e.cacheWriteCost, e.reasoning, e.toolCall, e.structuredOutput, e.attachment, e.modalities, e.reasoningType, e.reasoningLevels, e.reasoningLimits, e.source, time.Now().UTC(), existingID)
		return err
	}
	id := uuid.NewString()
	_, err = s.db.Exec(db.Q(`INSERT INTO provider_models(id, provider_id, model_id, display_name, owned_by, context_window, max_output, input_cost, output_cost, cache_read_cost, cache_write_cost, reasoning, tool_call, structured_output, attachment, modalities, reasoning_type, reasoning_levels, reasoning_output_limits, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`),
		id, p.ID, m.ID, displayName, m.OwnedBy, e.ctx, e.maxOut, e.inputCost, e.outputCost, e.cacheReadCost, e.cacheWriteCost, e.reasoning, e.toolCall, e.structuredOutput, e.attachment, e.modalities, e.reasoningType, e.reasoningLevels, e.reasoningLimits, e.source, time.Now().UTC(), time.Now().UTC())
	return err
}

// enrichment is the catalog-derived detail attached to a provider model at
// discovery/enrich time.
type enrichment struct {
	ctx, maxOut                    int
	inputCost, outputCost          float64
	cacheReadCost, cacheWriteCost  float64
	reasoning, toolCall            bool
	structuredOutput, attachment   bool
	modalities                     string
	reasoningType, reasoningLevels string
	reasoningLimits, source        string
}

// resolveEnrichment merges provider-declared metadata over the models.dev
// catalog enrichment. Fields the provider's own /models entry declares win
// per-field — including a declared $0.00 price, which pointer semantics
// distinguish from "not declared"; undeclared fields keep catalog values.
// Source records the authoritative origin: "provider" when the listing
// carried usable detail, else the catalog outcome ("enriched" /
// "enriched-wildcard" / "discovered").
func (s *Service) resolveEnrichment(m rawModel) enrichment {
	e := s.enrichFor(m.ID)
	if m.Meta == nil {
		return e
	}
	pm := m.Meta
	if pm.ContextWindow > 0 {
		e.ctx = pm.ContextWindow
	}
	if pm.MaxOutput > 0 {
		e.maxOut = pm.MaxOutput
	}
	if pm.InputCost != nil {
		e.inputCost = *pm.InputCost
	}
	if pm.OutputCost != nil {
		e.outputCost = *pm.OutputCost
	}
	if pm.CacheReadCost != nil {
		e.cacheReadCost = *pm.CacheReadCost
	}
	if pm.CacheWriteCost != nil {
		e.cacheWriteCost = *pm.CacheWriteCost
	}
	if pm.Reasoning != nil {
		e.reasoning = *pm.Reasoning
	}
	if pm.ToolCall != nil {
		e.toolCall = *pm.ToolCall
	}
	if pm.StructuredOutput != nil {
		e.structuredOutput = *pm.StructuredOutput
	}
	if pm.Attachment != nil {
		e.attachment = *pm.Attachment
	}
	if pm.Modalities != "" {
		e.modalities = pm.Modalities
	}
	e.source = "provider"
	return e
}

// enrichFor resolves catalog detail for any upstream model ID, including
// reseller-tagged ones ("[aws] grok-4.6"). Exact catalog hits record source
// "enriched"; approximate wildcard hits record "enriched-wildcard" so
// estimated pricing stays distinguishable; misses stay "discovered".
func (s *Service) enrichFor(modelID string) enrichment {
	e := enrichment{source: "discovered"}
	if s.catalogStore == nil {
		return e
	}
	cm, kind, err := s.catalogStore.FindBestMatch(modelID)
	if err != nil {
		return e
	}
	e.ctx, e.maxOut = cm.ContextWindow, cm.MaxOutput
	e.inputCost, e.outputCost = cm.InputCost, cm.OutputCost
	e.cacheReadCost, e.cacheWriteCost = cm.CacheReadCost, cm.CacheWriteCost
	e.reasoning, e.toolCall, e.structuredOutput, e.attachment = cm.Reasoning, cm.ToolCall, cm.StructuredOutput, cm.Attachment
	e.modalities = cm.Modalities
	e.reasoningType, e.reasoningLevels, e.reasoningLimits = cm.ReasoningType, cm.ReasoningLevels, cm.ReasoningOutputLimits
	e.source = "enriched"
	if kind == "wildcard" {
		e.source = "enriched-wildcard"
	}
	return e
}

// List returns provider_models with provider join, filtered
func (s *Service) List(providerID, q string) ([]models.ProviderModel, error) {
	where := "1=1"
	args := []interface{}{}
	if providerID != "" {
		where += " AND pm.provider_id = ?"
		args = append(args, providerID)
	}
	if q != "" {
		where += " AND (pm.model_id LIKE ? OR pm.display_name LIKE ? OR p.name LIKE ? OR (p.name || '/' || pm.model_id) LIKE ?)"
		like := "%" + q + "%"
		args = append(args, like, like, like, like)
	}
	rows, err := s.db.Query(db.Q(`SELECT pm.id, pm.provider_id, pm.model_id, pm.display_name, pm.owned_by, pm.context_window, pm.max_output, pm.input_cost, pm.output_cost, pm.cache_read_cost, pm.cache_write_cost, pm.reasoning, pm.tool_call, pm.structured_output, pm.attachment, pm.modalities, pm.source, pm.created_at, pm.updated_at, pm.reasoning_type, pm.reasoning_levels, pm.reasoning_output_limits, p.name FROM provider_models pm JOIN providers p ON p.id=pm.provider_id WHERE `+where+` ORDER BY p.name ASC, pm.model_id ASC LIMIT 500`), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []models.ProviderModel
	for rows.Next() {
		var pm models.ProviderModel
		var cc, cw sql.NullFloat64
		var rn, tl, so, at sql.NullBool
		var rt, rl, rol sql.NullString
		var provName string
		if err := rows.Scan(&pm.ID, &pm.ProviderID, &pm.ModelID, &pm.DisplayName, &pm.OwnedBy, &pm.ContextWindow, &pm.MaxOutput, &pm.InputCost, &pm.OutputCost, &cc, &cw, &rn, &tl, &so, &at, &pm.Modalities, &pm.Source, &pm.CreatedAt, &pm.UpdatedAt, &rt, &rl, &rol, &provName); err != nil {
			continue
		}
		if cc.Valid {
			pm.CacheReadCost = cc.Float64
		}
		if cw.Valid {
			pm.CacheWriteCost = cw.Float64
		}
		if rn.Valid {
			pm.Reasoning = rn.Bool
		}
		if tl.Valid {
			pm.ToolCall = tl.Bool
		}
		if so.Valid {
			pm.StructuredOutput = so.Bool
		}
		if at.Valid {
			pm.Attachment = at.Bool
		}
		if rt.Valid {
			pm.ReasoningType = rt.String
		}
		if rl.Valid {
			pm.ReasoningLevels = rl.String
		}
		if rol.Valid {
			pm.ReasoningOutputLimits = rol.String
		}
		pm.ProviderName = provName
		out = append(out, pm)
	}
	return out, nil
}

func (s *Service) Enrich(providerModelID string) error {
	var providerID, modelID string
	err := s.db.QueryRow(db.Q(`SELECT provider_id, model_id FROM provider_models WHERE id=?`), providerModelID).Scan(&providerID, &modelID)
	if err != nil {
		return err
	}
	// OAuth providers with bespoke enrichment: the models.dev catalog has no
	// entries for their ids, so the generic path below would wipe the
	// discovery enrichment (context, costs, reasoning levels) to zeros.
	// Re-derive from the provider source instead.
	if s.providerStore != nil {
		if p, perr := s.providerStore.GetByID(providerID); perr == nil && p != nil {
			switch p.Type {
			case models.ProviderDevin:
				if err := s.enrichDevinRow(p, providerModelID, modelID); err == nil {
					return nil
				}
			case models.ProviderAntigravity:
				if err := s.enrichAntigravityRow(providerModelID, modelID); err == nil {
					return nil
				}
			}
		}
	}
	e := s.enrichFor(modelID)
	_, err = s.db.Exec(db.Q(`UPDATE provider_models SET context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, structured_output=?, attachment=?, modalities=?, reasoning_type=?, reasoning_levels=?, reasoning_output_limits=?, source=?, updated_at=? WHERE id=?`),
		e.ctx, e.maxOut, e.inputCost, e.outputCost, e.cacheReadCost, e.cacheWriteCost, e.reasoning, e.toolCall, e.structuredOutput, e.attachment, e.modalities, e.reasoningType, e.reasoningLevels, e.reasoningLimits, e.source, time.Now().UTC(), providerModelID)
	return err
}

func (s *Service) UpdateManual(id string, upd models.ProviderModel) error {
	_, err := s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, reasoning=?, tool_call=?, structured_output=?, attachment=?, modalities=?, reasoning_type=?, reasoning_levels=?, reasoning_output_limits=?, source=?, updated_at=? WHERE id=?`),
		upd.DisplayName, upd.OwnedBy, upd.ContextWindow, upd.MaxOutput, upd.InputCost, upd.OutputCost, upd.Reasoning, upd.ToolCall, upd.StructuredOutput, upd.Attachment, upd.Modalities, upd.ReasoningType, upd.ReasoningLevels, upd.ReasoningOutputLimits, "manual", time.Now().UTC(), id)
	return err
}

func (s *Service) Delete(id string) error {
	var providerID, modelID string
	err := s.db.QueryRow(db.Q(`SELECT provider_id, model_id FROM provider_models WHERE id=?`), id).Scan(&providerID, &modelID)
	if err == sql.ErrNoRows {
		return nil
	}
	if err != nil {
		return err
	}
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	// Snapshot before the delete: the recycling bin restores this exact row.
	snap, ok, err := loadSnapshot(tx, id)
	if err != nil {
		tx.Rollback()
		return err
	}
	if _, err := tx.Exec(db.Q(`DELETE FROM provider_models WHERE id=?`), id); err != nil {
		tx.Rollback()
		return err
	}
	var raw []byte
	if ok {
		raw, err = json.Marshal(snap)
		if err != nil {
			tx.Rollback()
			return err
		}
	}
	if err := insertExclusion(tx, providerID, modelID, raw); err != nil {
		tx.Rollback()
		return err
	}
	return tx.Commit()
}

func (s *Service) AddManual(providerID, modelID string, upd models.ProviderModel) (string, error) {
	id := uuid.NewString()
	ctx, maxOut := upd.ContextWindow, upd.MaxOutput
	if ctx == 0 && maxOut == 0 && s.catalogStore != nil {
		if cm, _, err := s.catalogStore.FindBestMatch(modelID); err == nil {
			ctx = cm.ContextWindow
			maxOut = cm.MaxOutput
		}
	}
	tx, err := s.db.Begin()
	if err != nil {
		return "", err
	}
	// A manual add is the only way back in after a removal.
	if _, err := tx.Exec(db.Q(`DELETE FROM provider_model_exclusions WHERE provider_id=? AND model_id=?`), providerID, modelID); err != nil {
		tx.Rollback()
		return "", err
	}
	_, err = tx.Exec(db.Q(`INSERT INTO provider_models(id, provider_id, model_id, display_name, owned_by, context_window, max_output, input_cost, output_cost, cache_read_cost, cache_write_cost, reasoning, tool_call, structured_output, attachment, modalities, reasoning_type, reasoning_levels, reasoning_output_limits, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`),
		id, providerID, modelID, upd.DisplayName, upd.OwnedBy, ctx, maxOut, upd.InputCost, upd.OutputCost, upd.CacheReadCost, upd.CacheWriteCost, upd.Reasoning, upd.ToolCall, upd.StructuredOutput, upd.Attachment, upd.Modalities, upd.ReasoningType, upd.ReasoningLevels, upd.ReasoningOutputLimits, "manual", time.Now().UTC(), time.Now().UTC())
	if err != nil {
		tx.Rollback()
		return "", err
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	return id, nil
}

// ListExcluded returns the recycling bin: models an operator removed, which
// discovery will not reinsert. Housekeeping exclusions (pruned variants) have
// no snapshot and are not shown.
func (s *Service) ListExcluded(providerID, q string) ([]models.ExcludedModel, error) {
	where := "e.snapshot IS NOT NULL"
	args := []interface{}{}
	if providerID != "" {
		where += " AND e.provider_id = ?"
		args = append(args, providerID)
	}
	if q != "" {
		where += " AND (e.model_id LIKE ? OR p.name LIKE ? OR (p.name || '/' || e.model_id) LIKE ? OR e.snapshot LIKE ?)"
		like := "%" + q + "%"
		args = append(args, like, like, like, like)
	}
	rows, err := s.db.Query(db.Q(`SELECT e.id, e.provider_id, e.model_id, e.created_at, e.snapshot, p.name FROM provider_model_exclusions e JOIN providers p ON p.id=e.provider_id WHERE `+where+` ORDER BY e.created_at DESC LIMIT 500`), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []models.ExcludedModel
	for rows.Next() {
		var em models.ExcludedModel
		var raw sql.NullString
		if err := rows.Scan(&em.ID, &em.ProviderID, &em.ModelID, &em.RemovedAt, &raw, &em.ProviderName); err != nil {
			continue
		}
		if raw.Valid && raw.String != "" {
			var snap modelSnapshot
			if json.Unmarshal([]byte(raw.String), &snap) == nil {
				pm := models.ProviderModel{
					ID: snap.ID, ProviderID: snap.ProviderID, ProviderName: em.ProviderName,
					ModelID: snap.ModelID, DisplayName: snap.DisplayName, OwnedBy: snap.OwnedBy,
					ContextWindow: snap.ContextWindow, MaxOutput: snap.MaxOutput,
					InputCost: snap.InputCost, OutputCost: snap.OutputCost,
					CacheReadCost: snap.CacheReadCost, CacheWriteCost: snap.CacheWriteCost,
					Reasoning: snap.Reasoning, ToolCall: snap.ToolCall,
					StructuredOutput: snap.StructuredOutput, Attachment: snap.Attachment,
					Modalities: snap.Modalities, Source: snap.Source,
					CreatedAt: snap.CreatedAt, UpdatedAt: snap.UpdatedAt,
					ReasoningType: snap.ReasoningType, ReasoningLevels: snap.ReasoningLevels,
					ReasoningOutputLimits: snap.ReasoningOutputLimits,
				}
				em.Snapshot = &pm
			}
		}
		out = append(out, em)
	}
	return out, nil
}

// Restore puts a recycling-bin model back. A snapshotted row returns exactly
// as it was removed; a legacy exclusion (no snapshot) returns as a plain
// discovered model that the next discovery run will enrich. Restoring clears
// the exclusion, so discovery treats the model normally again.
func (s *Service) Restore(exclusionID string) (string, error) {
	var providerID, modelID string
	var raw sql.NullString
	err := s.db.QueryRow(db.Q(`SELECT provider_id, model_id, snapshot FROM provider_model_exclusions WHERE id=?`), exclusionID).Scan(&providerID, &modelID, &raw)
	if err != nil {
		return "", err
	}
	tx, err := s.db.Begin()
	if err != nil {
		return "", err
	}
	var newID string
	if raw.Valid && strings.TrimSpace(raw.String) != "" {
		var snap modelSnapshot
		if err := json.Unmarshal([]byte(raw.String), &snap); err != nil {
			tx.Rollback()
			return "", fmt.Errorf("recycling bin snapshot is corrupt")
		}
		newID = uuid.NewString()
		now := time.Now().UTC()
		_, err = tx.Exec(db.Q(`INSERT INTO provider_models(id, provider_id, model_id, display_name, owned_by, context_window, max_output, input_cost, output_cost, cache_read_cost, cache_write_cost, reasoning, tool_call, structured_output, attachment, modalities, reasoning_type, reasoning_levels, reasoning_output_limits, reasoning_routing, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`),
			newID, providerID, modelID, snap.DisplayName, snap.OwnedBy, snap.ContextWindow, snap.MaxOutput, snap.InputCost, snap.OutputCost, snap.CacheReadCost, snap.CacheWriteCost, snap.Reasoning, snap.ToolCall, snap.StructuredOutput, snap.Attachment, snap.Modalities, snap.ReasoningType, snap.ReasoningLevels, snap.ReasoningOutputLimits, snap.ReasoningRouting, snap.Source, snap.CreatedAt, now)
	} else {
		newID, err = s.restoreBare(tx, providerID, modelID)
	}
	if err != nil {
		tx.Rollback()
		return "", err
	}
	if _, err := tx.Exec(db.Q(`DELETE FROM provider_model_exclusions WHERE id=?`), exclusionID); err != nil {
		tx.Rollback()
		return "", err
	}
	if err := tx.Commit(); err != nil {
		return "", err
	}
	if s.Cache != nil {
		s.Cache.Invalidate("models:")
	}
	return newID, nil
}

// restoreBare reinserts a model that was excluded before snapshots existed.
func (s *Service) restoreBare(tx *sql.Tx, providerID, modelID string) (string, error) {
	id := uuid.NewString()
	now := time.Now().UTC()
	_, err := tx.Exec(db.Q(`INSERT INTO provider_models(id, provider_id, model_id, display_name, source, created_at, updated_at) VALUES(?,?,?,?,?,?,?)`),
		id, providerID, modelID, modelID, "discovered", now, now)
	return id, err
}

func (s *Service) DiscoverAll() (int, error) {
	// Materialize provider IDs FIRST and close the rows before issuing any
	// per-provider queries: SQLite runs with MaxOpenConns(1), so iterating an
	// open result set while Discover() needs a connection deadlocks the sole
	// connection and wedges the entire gateway.
	rows, err := s.db.Query(db.Q(`SELECT id FROM providers`))
	if err != nil {
		return 0, err
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err == nil {
			ids = append(ids, id)
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return 0, err
	}
	total := 0
	for _, id := range ids {
		if n, err := s.Discover(id); err == nil {
			total += n
		}
	}
	return total, nil
}

// isMultiProvider mirrors proxy.isMultiProtocolProvider without importing the
// proxy package (which would cycle). Keep the two in sync.
func isMultiProvider(p *models.Provider) bool {
	if p == nil {
		return false
	}
	base := strings.ToLower(strings.TrimSpace(p.BaseURL))
	name := strings.ToLower(strings.TrimSpace(p.Name))
	if strings.Contains(base, "opencode.ai/zen") || strings.Contains(base, "opencode.ai/go") {
		return true
	}
	for _, pre := range []string{"opencode-go", "opencode_go", "opencodego", "opencode-zen", "opencode_zen"} {
		if name == pre || strings.HasPrefix(name, pre+"/") || strings.HasPrefix(name, pre+"-") || strings.HasPrefix(name, pre+"_") {
			return true
		}
	}
	return false
}

// probeContextLimit asks the provider for its real context cap by sending a
// tiny completion with an absurd max_tokens; resellers that cap the model
// (kourier et al.) answer 400 with the leaked limit in the error text. When
// the upstream doesn't answer the question — auth errors, 404s, free-form
// errors — the probe returns 0 and catalog/listing values stand.
//
// Skips entirely for providers with no API key to probe with.
func (s *Service) probeContextLimit(p *models.Provider, apiKey, modelID string) int {
	if apiKey == "" || s.client == nil || p == nil {
		return 0
	}
	// Probe the cheapest endpoint dialect for this provider: OpenAI chat
	// completions for OpenAI-compatible, Anthropic messages for anthropic
	// providers (minimax uses the anthropic dialect).
	var path, body string
	var authHdr string
	if p.Type == models.ProviderAnthropic {
		path = strings.TrimRight(p.BaseURL, "/") + "/messages"
		if !strings.HasSuffix(p.BaseURL, "/v1") && !strings.Contains(p.BaseURL, "/v1/") {
			path = strings.TrimRight(p.BaseURL, "/") + "/v1/messages"
		}
		body = `{"model":"` + modelID + `","max_tokens":999999,"messages":[{"role":"user","content":"x"}]}`
		authHdr = "x-api-key"
	} else {
		path = strings.TrimRight(p.BaseURL, "/") + "/chat/completions"
		if !strings.Contains(p.BaseURL, "/v1") {
			path = strings.TrimRight(p.BaseURL, "/") + "/v1/chat/completions"
		}
		body = `{"model":"` + modelID + `","max_tokens":999999,"messages":[{"role":"user","content":"x"}]}`
		authHdr = "Authorization"
	}
	req, err := http.NewRequest("POST", path, strings.NewReader(body))
	if err != nil {
		return 0
	}
	if authHdr == "Authorization" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	} else {
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	// Only trust answers that came back fast and look like a real cap error;
	// 401/404/5xx mean the probe didn't reach a model validator.
	if resp.StatusCode != http.StatusBadRequest && resp.StatusCode != http.StatusUnprocessableEntity {
		return 0
	}
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
	return parseLeakedLimit(string(b))
}

// parseLeakedLimit extracts the provider's real token cap from a 400 error
// body. Handles the observed resellers' phrasing:
//
//	kourier (bifrost): "max_model_len=max_total_tokens=262144"
//	vllm/sglang:       "max_model_len", "max_tokens", "context length"
//	openrouter:        "context length", "max context"
func parseLeakedLimit(body string) int {
	re := regexp.MustCompile(`(?:max_model_len|max_total_tokens|context_length|context length|context window|max context)\s*[=:]\s*(\d+)`)
	if m := re.FindStringSubmatch(body); len(m) == 2 {
		if n, err := strconv.Atoi(m[1]); err == nil && n > 0 && n < 100_000_000 {
			return n
		}
	}
	return 0
}
