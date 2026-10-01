package discovery

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"ai-gateway/internal/db"
	"ai-gateway/internal/devin"
	"ai-gateway/internal/models"

	"github.com/google/uuid"
)

// discoverDevin seeds provider_models from the static fallback catalog, then
// merges live GetCliModelConfigs results when OAuth is connected.
//
// Devin advertises one wire config per (model, reasoning level) as
// "<model>-<level>" (e.g. "swe-2-max"). Variants collapse to a single base
// row carrying the observed reasoning levels; the proxy routes an explicit
// reasoning_effort back to the suffixed wire id. Storing each variant as its
// own row defeats enrichment (no catalog entry matches "swe-2-max") and
// leaves reasoning_levels empty behind a false source="enriched".
func (s *Service) discoverDevin(p *models.Provider) (int, error) {
	groups := s.devinGroups(p)
	count := 0
	for _, g := range groups {
		if err := s.upsertDevin(p, g); err == nil {
			count++
		}
	}
	s.pruneDevinVariants(p.ID, groups)
	if s.Cache != nil && count > 0 {
		s.Cache.Invalidate("models:")
	}
	if count == 0 {
		return 0, fmt.Errorf("no models discovered (check provider base_url and key)")
	}
	return count, nil
}

// devinGroups builds the collapsed base-model set: static fallback seeds
// overlaid with live GetCliModelConfigs variants. Live wins for
// display/context/costs; the observed level set replaces the seed default
// (CollapseModels never returns an empty set). Routing adopts
// server-declared or wire-observed maps; the seed's nil (suffix convention)
// survives bare-only live groups.
func (s *Service) devinGroups(p *models.Provider) map[string]devin.CollapsedModel {
	groups := map[string]devin.CollapsedModel{}
	for _, m := range devin.PublicModels {
		groups[m.ID] = devin.CollapsedModel{
			ID: m.ID, Name: m.Name,
			ContextWindow: m.ContextWindow, MaxTokens: m.MaxTokens,
			ImageInput: true, Reasoning: true, ReasoningType: "effort",
			ToolCalls: true, Levels: devin.DefaultReasoningLevels(),
		}
	}
	for _, g := range devin.CollapseModels(s.fetchDevinLive(p)) {
		prev, ok := groups[g.ID]
		if !ok {
			groups[g.ID] = g
			continue
		}
		prev.Name = g.Name
		if g.ContextWindow > prev.ContextWindow {
			prev.ContextWindow = g.ContextWindow
		}
		if g.MaxTokens > prev.MaxTokens {
			prev.MaxTokens = g.MaxTokens
		}
		prev.ImageInput = prev.ImageInput || g.ImageInput
		prev.Reasoning = g.Reasoning
		if g.ReasoningType != "" {
			prev.ReasoningType = g.ReasoningType
		}
		prev.ToolCalls = g.ToolCalls
		prev.InputCost, prev.OutputCost, prev.CacheReadCost = g.InputCost, g.OutputCost, g.CacheReadCost
		prev.Levels = g.Levels
		if len(g.Routing) > 0 {
			prev.Routing = g.Routing
		}
		groups[g.ID] = prev
	}
	return groups
}

func (s *Service) upsertDevin(p *models.Provider, g devin.CollapsedModel) error {
	// Operator overrides survive rediscovery.
	if s.isManual(p.ID, g.ID) {
		return nil
	}
	if err := s.upsert(p, rawModel{ID: g.ID, OwnedBy: "devin"}, "", nil); err != nil {
		return err
	}
	return s.writeDevinRow(p.ID, g.ID, g)
}

// devinRowArgs renders a collapsed model as provider_models columns.
// Costs are the server-declared per-million-token rates (informational:
// Devin bills via seat/quota, so proxy token costs stay zero).
// reasoning_type/levels/routing are stored explicitly — the generic upsert
// above cannot enrich devin ids from the catalog, and without this the rows
// claim source "enriched" with empty reasoning metadata.
func devinRowArgs(g devin.CollapsedModel) []any {
	levels, _ := json.Marshal(g.Levels)
	if len(g.Levels) == 0 || string(levels) == "null" {
		levels = []byte("[]")
	}
	// Nil routing (legacy rows, fallback seeds) selects the suffix
	// convention at request time; store SQL NULL, never "null".
	var routing any
	if g.Routing != nil {
		rb, _ := json.Marshal(g.Routing)
		routing = string(rb)
	}
	rType := g.ReasoningType
	if rType == "" {
		rType = "effort"
	}
	return []any{
		g.Name, "devin", g.ContextWindow, g.MaxTokens,
		g.InputCost, g.OutputCost, g.CacheReadCost, 0, g.Reasoning, g.ToolCalls, g.ImageInput,
		rType, string(levels), "{}", routing, "enriched", time.Now().UTC(),
	}
}

func (s *Service) writeDevinRow(providerID, modelID string, g devin.CollapsedModel) error {
	args := append(devinRowArgs(g), providerID, modelID)
	_, err := s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, attachment=?, reasoning_type=?, reasoning_levels=?, reasoning_output_limits=?, reasoning_routing=?, source=?, updated_at=? WHERE provider_id=? AND model_id=?`), args...)
	return err
}

// enrichDevinRow re-applies collapsed enrichment to a single row for the
// per-model Enrich endpoint (which otherwise resolves the models.dev catalog
// — with no devin entries — and wipes the row to zeros). Variant rows keep
// their model_id but receive their base's metadata, staying functional until
// the next discovery prunes them. Unknown ids report sql.ErrNoRows so the
// caller can fall back to the catalog path.
func (s *Service) enrichDevinRow(p *models.Provider, rowID, modelID string) error {
	g, ok := s.devinGroups(p)[devin.CollapseReasoningVariant(modelID)]
	if !ok {
		return sql.ErrNoRows
	}
	args := append(devinRowArgs(g), rowID)
	_, err := s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, attachment=?, reasoning_type=?, reasoning_levels=?, reasoning_output_limits=?, reasoning_routing=?, source=?, updated_at=? WHERE id=?`), args...)
	return err
}

// isManual reports whether a provider model row carries an operator override
// that rediscovery must not clobber.
func (s *Service) isManual(providerID, modelID string) bool {
	var src sql.NullString
	if err := s.db.QueryRow(db.Q(`SELECT source FROM provider_models WHERE provider_id=? AND model_id=?`), providerID, modelID).Scan(&src); err == nil && src.Valid && src.String == "manual" {
		return true
	}
	return false
}

// isExcluded reports whether the operator removed this model from the
// provider. Discovery must not reinsert it; restoring from the recycling bin
// or a manual add clears the mark.
func (s *Service) isExcluded(providerID, modelID string) bool {
	var n int
	err := s.db.QueryRow(db.Q(`SELECT COUNT(*) FROM provider_model_exclusions WHERE provider_id=? AND model_id=?`), providerID, modelID).Scan(&n)
	return err == nil && n > 0
}

// modelSnapshot is the provider_models row frozen into the recycling bin.
// Traffic stats are derived at read time and are not part of the row.
type modelSnapshot struct {
	ID                    string    `json:"id"`
	ProviderID            string    `json:"provider_id"`
	ModelID               string    `json:"model_id"`
	DisplayName           string    `json:"display_name"`
	OwnedBy               string    `json:"owned_by"`
	ContextWindow         int       `json:"context_window"`
	MaxOutput             int       `json:"max_output"`
	InputCost             float64   `json:"input_cost"`
	OutputCost            float64   `json:"output_cost"`
	CacheReadCost         float64   `json:"cache_read_cost"`
	CacheWriteCost        float64   `json:"cache_write_cost"`
	Reasoning             bool      `json:"reasoning"`
	ToolCall              bool      `json:"tool_call"`
	StructuredOutput      bool      `json:"structured_output"`
	Attachment            bool      `json:"attachment"`
	Modalities            string    `json:"modalities"`
	ReasoningType         string    `json:"reasoning_type"`
	ReasoningLevels       string    `json:"reasoning_levels"`
	ReasoningOutputLimits string    `json:"reasoning_output_limits"`
	ReasoningRouting      *string   `json:"reasoning_routing"`
	Source                string    `json:"source"`
	CreatedAt             time.Time `json:"created_at"`
	UpdatedAt             time.Time `json:"updated_at"`
}

// loadSnapshot reads the live provider_models row that a removal is about to
// drop. A missing row (already gone) yields ok=false and no error.
func loadSnapshot(q dbQuerier, id string) (modelSnapshot, bool, error) {
	var snap modelSnapshot
	var cc, cw sql.NullFloat64
	var rn, tl, so, at sql.NullBool
	var mods, rt, rl, rol, routing sql.NullString
	err := q.QueryRow(db.Q(`SELECT id, provider_id, model_id, COALESCE(display_name,''), COALESCE(owned_by,''), COALESCE(context_window,0), COALESCE(max_output,0), COALESCE(input_cost,0), COALESCE(output_cost,0), cache_read_cost, cache_write_cost, reasoning, tool_call, structured_output, attachment, modalities, reasoning_type, reasoning_levels, reasoning_output_limits, reasoning_routing, COALESCE(source,''), created_at, updated_at FROM provider_models WHERE id=?`), id).
		Scan(&snap.ID, &snap.ProviderID, &snap.ModelID, &snap.DisplayName, &snap.OwnedBy, &snap.ContextWindow, &snap.MaxOutput, &snap.InputCost, &snap.OutputCost, &cc, &cw, &rn, &tl, &so, &at, &mods, &rt, &rl, &rol, &routing, &snap.Source, &snap.CreatedAt, &snap.UpdatedAt)
	if err == sql.ErrNoRows {
		return modelSnapshot{}, false, nil
	}
	if err != nil {
		return modelSnapshot{}, false, err
	}
	if cc.Valid {
		snap.CacheReadCost = cc.Float64
	}
	if cw.Valid {
		snap.CacheWriteCost = cw.Float64
	}
	snap.Reasoning, snap.ToolCall, snap.StructuredOutput, snap.Attachment = rn.Bool, tl.Bool, so.Bool, at.Bool
	snap.Modalities, snap.ReasoningType, snap.ReasoningLevels, snap.ReasoningOutputLimits = mods.String, rt.String, rl.String, rol.String
	if routing.Valid {
		snap.ReasoningRouting = &routing.String
	}
	return snap, true, nil
}

// dbQuerier is the subset of *sql.DB and *sql.Tx the snapshot helpers need.
type dbQuerier interface {
	QueryRow(query string, args ...any) *sql.Row
	Exec(query string, args ...any) (sql.Result, error)
}

// insertExclusion writes the recycling-bin row. snapshotJSON is nil for
// housekeeping removals (pruned variants) that should not surface in the bin.
func insertExclusion(q dbQuerier, providerID, modelID string, snapshotJSON []byte) error {
	_, err := q.Exec(db.Q(`INSERT INTO provider_model_exclusions(id, provider_id, model_id, created_at, snapshot) VALUES(?,?,?,?,?)`+db.UpsertEnd([]string{"provider_id", "model_id"}, []string{"created_at", "snapshot"})),
		uuid.NewString(), providerID, modelID, time.Now().UTC(), snapshotJSON)
	return err
}

// exclude records a removal so discovery will not reinsert the model. Used
// where there is no surrounding transaction (variant pruning). Pruned
// variants are housekeeping, not an operator choice, so they carry no
// snapshot and stay out of the recycling bin.
func (s *Service) exclude(providerID, modelID string) error {
	return insertExclusion(s.db, providerID, modelID, nil)
}

// pruneDevinVariants removes pre-collapse per-variant rows (e.g. "swe-2-max")
// now represented by their collapsed base row. Scoped to variant ids that
// resolve into a stored base; manual rows are never touched.
func (s *Service) pruneDevinVariants(providerID string, bases map[string]devin.CollapsedModel) {
	rows, err := s.db.Query(db.Q(`SELECT id, model_id, source FROM provider_models WHERE provider_id=?`), providerID)
	if err != nil {
		return
	}
	var stale []struct{ id, modelID string }
	for rows.Next() {
		var id, modelID string
		var source sql.NullString
		if err := rows.Scan(&id, &modelID, &source); err != nil {
			continue
		}
		if source.Valid && source.String == "manual" {
			continue
		}
		base := devin.CollapseReasoningVariant(modelID)
		if base == modelID {
			continue
		}
		if _, ok := bases[base]; !ok {
			continue
		}
		stale = append(stale, struct{ id, modelID string }{id, modelID})
	}
	// Close before any further query: SQLite runs with MaxOpenConns(1), so
	// writing while this result set is open deadlocks the sole connection.
	rows.Close()
	for _, row := range stale {
		// A removed variant must stay removed: pruning it without an exclusion
		// would let the next discovery reinsert it under its collapsed base.
		if err := s.exclude(providerID, row.modelID); err != nil {
			continue
		}
		_, _ = s.db.Exec(db.Q(`DELETE FROM provider_models WHERE id=?`), row.id)
	}
}

// fetchDevinLive returns live model configs with a fresh OAuth session token.
func (s *Service) fetchDevinLive(p *models.Provider) []devin.DiscoveredModel {
	if s.providerStore == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	access, _, _, err := s.providerStore.EnsureFreshAccess(ctx, p, s.client)
	if err != nil || access == "" {
		return nil
	}
	base := p.BaseURL
	if base == "" {
		base = devin.Host()
	}
	live, err := devin.FetchModels(access, base, s.client)
	if err != nil {
		return nil
	}
	return live
}
