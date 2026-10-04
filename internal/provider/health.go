package provider

import (
	"context"
	"database/sql"
	"net/http"
	"strings"
	"time"

	"ai-gateway/internal/devin"

	gatewaydb "ai-gateway/internal/db"

	"ai-gateway/internal/models"

	"github.com/rs/zerolog/log"
)

func checkCtx() context.Context {
	return context.Background()
}

func StartHealthChecker(db *sql.DB, store *Store, interval time.Duration) {
	if interval <= 0 {
		interval = 5 * time.Minute
	}
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		// run once at start after short delay
		time.Sleep(10 * time.Second)
		checkAll(db, store)
		for range ticker.C {
			checkAll(db, store)
		}
	}()
}

func checkAll(db *sql.DB, store *Store) {
	providers, err := store.List()
	if err != nil {
		log.Error().Err(err).Msg("health: list providers failed")
		return
	}
	client := &http.Client{Timeout: 5 * time.Second}
	for _, p := range providers {
		status := "unknown"
		msg := ""
		target := p.BaseURL
		// Antigravity is OAuth-backed: health = token refreshability, not /models.
		if p.Type == "antigravity" {
			if tok, _, err := store.OAuthTokens(&p); err != nil || tok == nil || tok.Refresh == "" {
				status = "down"
				msg = "oauth not connected — connect in Providers"
			} else if _, _, _, rerr := store.EnsureFreshAccess(checkCtx(), &p, client); rerr != nil {
				status = "down"
				msg = "oauth refresh failed — reconnect"
			} else {
				status = "up"
				msg = "OK (oauth)"
			}
			if _, err := db.Exec(gatewaydb.Q(`UPDATE providers SET health_status=?, last_health=? WHERE id=?`), status, msg, p.ID); err != nil {
				log.Error().Err(err).Str("provider", p.Name).Msg("health update failed")
			}
			continue
		}
		// Devin is OAuth-backed: health = session token + user-JWT probe.
		if p.Type == "devin" {
			if tok, _, err := store.OAuthTokens(&p); err != nil || tok == nil || tok.Refresh == "" {
				status = "down"
				msg = "oauth not connected — connect in Providers"
			} else if access, _, _, rerr := store.EnsureFreshAccess(checkCtx(), &p, client); rerr != nil {
				status = "down"
				msg = "oauth refresh failed — reconnect"
			} else if _, _, derr := devin.GetUserJWT(access, p.BaseURL, client); derr != nil {
				status = "down"
				msg = "devin auth failed — reconnect"
			} else {
				status = "up"
				msg = "OK (oauth)"
			}
			if _, err := db.Exec(gatewaydb.Q(`UPDATE providers SET health_status=?, last_health=? WHERE id=?`), status, msg, p.ID); err != nil {
				log.Error().Err(err).Str("provider", p.Name).Msg("health update failed")
			}
			continue
		}
		switch p.Type {
		case "anthropic":
			// anthropic: try to ping /v1/models with x-api-key if possible, otherwise mark unknown
			key, err := store.DecryptKey(&p)
			if err == nil {
				baseModels := strings.TrimRight(target, "/")
				var targetModels string
				switch {
				case strings.HasSuffix(baseModels, "/v1/models"):
					targetModels = baseModels
				case strings.Contains(baseModels, "/v1"):
					// Base already carries a version prefix (e.g. https://ckff.dev/v1):
					// appending another /v1 would build /v1/v1/models (404).
					targetModels = baseModels + "/models"
				default:
					targetModels = baseModels + "/v1/models"
				}
				req, _ := http.NewRequest("GET", targetModels, nil)
				req.Header.Set("User-Agent", models.GatewayUserAgent)
				req.Header.Set("x-api-key", key)
				req.Header.Set("anthropic-version", "2023-06-01")
				resp, err := client.Do(req)
				if err != nil {
					status = "down"
					msg = err.Error()
				} else {
					resp.Body.Close()
					if resp.StatusCode == 200 || resp.StatusCode == 401 || resp.StatusCode == 403 {
						status = "up"
						msg = http.StatusText(resp.StatusCode)
					} else if resp.StatusCode >= 500 {
						status = "down"
						msg = http.StatusText(resp.StatusCode)
					} else {
						status = "up"
						msg = http.StatusText(resp.StatusCode)
					}
				}
			} else {
				status = "unknown"
				msg = "no key"
			}
		case "azure":
			key, err := store.DecryptKey(&p)
			if err != nil {
				status = "down"
				msg = "decrypt failed"
			} else {
				// Azure OpenAI: GET {baseURL}/models?api-version=2024-02-01 with api-key
				base := strings.TrimRight(target, "/")
				urls := []string{base + "/models?api-version=2024-02-01", base + "/models"}
				if !strings.Contains(base, "api-version") && !strings.Contains(target, "/v1") {
					urls = append(urls, base+"/v1/models?api-version=2024-02-01")
				}
				success := false
				for _, u := range urls {
					req, _ := http.NewRequest("GET", u, nil)
					req.Header.Set("User-Agent", models.GatewayUserAgent)
					req.Header.Set("api-key", key)
					resp, err := client.Do(req)
					if err != nil {
						status = "down"
						msg = err.Error()
						continue
					}
					resp.Body.Close()
					if resp.StatusCode == 200 {
						status = "up"
						msg = "OK"
						success = true
						break
					} else if resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 429 {
						status = "up"
						msg = http.StatusText(resp.StatusCode) + " (auth ok, reachable)"
						success = true
						break
					} else if resp.StatusCode >= 500 {
						status = "down"
						msg = http.StatusText(resp.StatusCode)
					} else if resp.StatusCode == 404 {
						status = "down"
						msg = "404 models not found (check base_url)"
					} else {
						status = "up"
						msg = http.StatusText(resp.StatusCode)
						success = true
						break
					}
				}
				if !success && status == "unknown" {
					status = "down"
					msg = "unreachable"
				}
			}
		default:
			key, err := store.DecryptKey(&p)
			if err != nil {
				status = "down"
				msg = "decrypt failed"
			} else {
				urls := []string{strings.TrimRight(target, "/") + "/models"}
				if !strings.Contains(target, "/v1") {
					urls = append(urls, strings.TrimRight(target, "/")+"/v1/models")
				}
				success := false
				for _, u := range urls {
					req, _ := http.NewRequest("GET", u, nil)
					req.Header.Set("User-Agent", models.GatewayUserAgent)
					req.Header.Set("Authorization", "Bearer "+key)
					resp, err := client.Do(req)
					if err != nil {
						status = "down"
						msg = err.Error()
						continue
					}
					resp.Body.Close()
					// Consider 200, 401, 403 as "up" (reachable), 5xx as down.
					// A 404 on the /models probe is NOT proof of an unhealthy
					// provider: many OpenAI-compatible relays don't implement
					// it while chat works fine. Treating 404 as "down" made
					// the LB skip working providers (potentially emptying a
					// whole routing group), so it's "unknown" now.
					if resp.StatusCode == 200 {
						status = "up"
						msg = "OK"
						success = true
						break
					} else if resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 429 {
						status = "up"
						msg = http.StatusText(resp.StatusCode) + " (auth ok, reachable)"
						success = true
						break
					} else if resp.StatusCode >= 500 {
						status = "down"
						msg = http.StatusText(resp.StatusCode)
					} else if resp.StatusCode == 404 || resp.StatusCode == 405 {
						status = "unknown"
						msg = "probe endpoint not implemented (404); provider kept"
						success = true
						break
					} else {
						status = "up"
						msg = http.StatusText(resp.StatusCode)
						success = true
						break
					}
				}
				// Multi-protocol providers (OpenCode Go/Zen) also serve
				// /v1/messages. If the OpenAI probe did not establish health,
				// try the Anthropic dialect before declaring down.
				if !success && isMultiProtocolBase(p.BaseURL, p.Name) {
					baseAnth := strings.TrimRight(target, "/")
					anthTarget := baseAnth + "/v1/models"
					if strings.HasSuffix(baseAnth, "/v1/models") {
						anthTarget = baseAnth
					} else if strings.Contains(baseAnth, "/v1") {
						anthTarget = baseAnth + "/models"
					}
					if req, _ := http.NewRequest("GET", anthTarget, nil); req != nil {
						req.Header.Set("User-Agent", models.GatewayUserAgent)
						req.Header.Set("x-api-key", key)
						req.Header.Set("anthropic-version", "2023-06-01")
						if resp, err := client.Do(req); err == nil {
							resp.Body.Close()
							if resp.StatusCode == 200 || resp.StatusCode == 401 || resp.StatusCode == 403 || resp.StatusCode == 429 {
								status = "up"
								msg = "OK (anthropic probe)"
								success = true
							}
						}
					}
				}
				if !success && status == "unknown" {
					status = "down"
					msg = "unreachable"
				}
			}
		}
		if _, err := db.Exec(gatewaydb.Q(`UPDATE providers SET health_status=?, last_health=? WHERE id=?`), status, msg, p.ID); err != nil {
			log.Error().Err(err).Str("provider", p.Name).Msg("health update failed")
		}
	}
}

// isMultiProtocolBase mirrors proxy.isMultiProtocolProvider without importing
// the proxy package. Keep the two in sync.
func isMultiProtocolBase(baseURL, name string) bool {
	base := strings.ToLower(strings.TrimSpace(baseURL))
	nm := strings.ToLower(strings.TrimSpace(name))
	if strings.Contains(base, "opencode.ai/zen") || strings.Contains(base, "opencode.ai/go") {
		return true
	}
	for _, pre := range []string{"opencode-go", "opencode_go", "opencodego", "opencode-zen", "opencode_zen"} {
		if nm == pre || strings.HasPrefix(nm, pre+"/") || strings.HasPrefix(nm, pre+"-") || strings.HasPrefix(nm, pre+"_") {
			return true
		}
	}
	return false
}
