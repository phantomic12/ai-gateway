package discovery

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"time"

	"ai-gateway/internal/antigravity"
	"ai-gateway/internal/db"
	"ai-gateway/internal/models"
)

// discoverAntigravity seeds provider_models from the static catalog, then tries
// a live fetchAvailableModels merge when OAuth is connected.
func (s *Service) discoverAntigravity(p *models.Provider) (int, error) {
	count := 0
	for _, m := range antigravity.PublicModels {
		rm := rawModel{ID: m.ID, OwnedBy: "antigravity"}
		if err := s.upsertAntigravity(p, m, rm); err == nil {
			count++
		}
	}
	if live := s.fetchAntigravityLive(p); len(live) > 0 {
		for _, id := range live {
			found := false
			for _, m := range antigravity.PublicModels {
				if m.ID == id {
					found = true
					break
				}
			}
			if !found {
				rm := rawModel{ID: id, OwnedBy: "antigravity"}
				pm := antigravity.PublicModel{ID: id, Name: id, ContextWindow: 200000, MaxTokens: 32000}
				if err := s.upsertAntigravity(p, pm, rm); err == nil {
					count++
				}
			}
		}
	}
	if s.Cache != nil && count > 0 {
		s.Cache.Invalidate("models:")
	}
	if count == 0 {
		return 0, context.DeadlineExceeded
	}
	return count, nil
}

func (s *Service) upsertAntigravity(p *models.Provider, pm antigravity.PublicModel, m rawModel) error {
	// Operator overrides survive rediscovery.
	if s.isManual(p.ID, m.ID) {
		return nil
	}
	// Reuse generic upsert path but force enriched costs from the static catalog.
	if err := s.upsert(p, m, "", nil); err != nil {
		return err
	}
	return s.writeAntigravityRow(p.ID, m.ID, pm)
}

func (s *Service) writeAntigravityRow(providerID, modelID string, pm antigravity.PublicModel) error {
	_, err := s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, attachment=?, source=?, updated_at=? WHERE provider_id=? AND model_id=?`),
		pm.Name, "antigravity", pm.ContextWindow, pm.MaxTokens, pm.InputCost, pm.OutputCost, pm.CacheReadCost, pm.CacheWriteCost, true, true, true, "enriched", time.Now().UTC(), providerID, modelID)
	return err
}

// enrichAntigravityRow re-applies static-catalog enrichment to a single row
// for the per-model Enrich endpoint (the models.dev catalog has no
// antigravity ids, so the generic path would wipe the row to zeros).
// Runtime variants resolve to their public base. Unknown ids report
// sql.ErrNoRows so the caller can fall back to the catalog path.
func (s *Service) enrichAntigravityRow(rowID, modelID string) error {
	base := collapseRuntime(modelID)
	for _, m := range antigravity.PublicModels {
		if m.ID != base {
			continue
		}
		_, err := s.db.Exec(db.Q(`UPDATE provider_models SET display_name=?, owned_by=?, context_window=?, max_output=?, input_cost=?, output_cost=?, cache_read_cost=?, cache_write_cost=?, reasoning=?, tool_call=?, attachment=?, source=?, updated_at=? WHERE id=?`),
			m.Name, "antigravity", m.ContextWindow, m.MaxTokens, m.InputCost, m.OutputCost, m.CacheReadCost, m.CacheWriteCost, true, true, true, "enriched", time.Now().UTC(), rowID)
		return err
	}
	return sql.ErrNoRows
}

// fetchAntigravityLive POSTs fetchAvailableModels with a fresh OAuth token.
// Returns runtime model ids (keys of data.models) or nil when disconnected.
func (s *Service) fetchAntigravityLive(p *models.Provider) []string {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	access, project, _, err := s.providerStore.EnsureFreshAccess(ctx, p, s.client)
	if err != nil || access == "" {
		return nil
	}
	if project == "" {
		project = antigravity.DefaultProjectID("")
	}
	body, _ := json.Marshal(map[string]any{"project": project})
	var last []string
	for _, ep := range antigravity.EndpointCandidates() {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(ep, "/")+"/v1internal:fetchAvailableModels", bytes.NewReader(body))
		if err != nil {
			continue
		}
		for k, v := range antigravity.Headers(access) {
			req.Header.Set(k, v)
		}
		resp, err := s.client.Do(req)
		if err != nil || resp.StatusCode != 200 {
			if resp != nil {
				resp.Body.Close()
			}
			continue
		}
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 5<<20))
		resp.Body.Close()
		var data map[string]any
		if json.Unmarshal(raw, &data) != nil {
			continue
		}
		if modelsMap, ok := data["models"].(map[string]any); ok {
			for id := range modelsMap {
				if strings.HasPrefix(id, "gemini-") || strings.HasPrefix(id, "claude-") || strings.HasPrefix(id, "gpt-oss-") {
					last = append(last, id)
				}
			}
		}
		if len(last) > 0 {
			// Merge across endpoints; keep going for sandbox-only ids.
			continue
		}
	}
	// Collapse runtime variants (gemini-3.8-flash-low/-medium/-high) to public ids.
	seen := map[string]bool{}
	var out []string
	for _, id := range last {
		pub := collapseRuntime(id)
		if !seen[pub] {
			seen[pub] = true
			out = append(out, pub)
		}
		// Also keep exact runtime ids that match public models.
		for _, m := range antigravity.PublicModels {
			if m.ID == id && !seen[id] {
				seen[id] = true
				out = append(out, id)
			}
		}
	}
	_ = models.ProviderAntigravity
	return out
}

func collapseRuntime(id string) string {
	for _, m := range antigravity.PublicModels {
		if id == m.ID || strings.HasPrefix(id, m.ID+"-") || strings.HasPrefix(id, m.ID) {
			return m.ID
		}
	}
	// Heuristic: strip trailing -low/-medium/-high/-extra-low/-thinking/-medium suffixes.
	for _, suffix := range []string{"-extra-low", "-thinking", "-medium", "-high", "-low"} {
		if strings.HasSuffix(id, suffix) {
			base := strings.TrimSuffix(id, suffix)
			for _, m := range antigravity.PublicModels {
				if base == m.ID {
					return m.ID
				}
			}
		}
	}
	return id
}
