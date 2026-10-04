package models

import "time"

// GatewayUserAgent is the User-Agent the gateway presents on every outbound
// request to a provider upstream. Go's net/http default ("Go-http-client/1.1")
// is rejected with HTTP 403 by the Cloudflare WAF in front of some upstreams
// (verified against inference-api.nousresearch.com — error 1010 "you have been
// blocked"), which silently broke model discovery, chat proxying, and health
// probes for those providers. Always set this explicitly on outbound requests.
const GatewayUserAgent = "ai-gateway/1.7.0"

type ProviderType string

const (
	ProviderOpenAI           ProviderType = "openai"
	ProviderAnthropic        ProviderType = "anthropic"
	ProviderAzure            ProviderType = "azure"
	ProviderOpenAICompatible ProviderType = "openai_compatible"
	ProviderAntigravity      ProviderType = "antigravity"
	ProviderDevin            ProviderType = "devin"
)

type Provider struct {
	ID           string       `json:"id"`
	Name         string       `json:"name"`
	Type         ProviderType `json:"type"`
	BaseURL      string       `json:"base_url"`
	APIKeyEnc    []byte       `json:"-"`                 // encrypted
	APIKey       string       `json:"api_key,omitempty"` // only on create, masked on read
	CreatedAt    time.Time    `json:"created_at"`
	LastHealth   *string      `json:"last_health,omitempty"`
	HealthStatus *string      `json:"health_status,omitempty"`
	OrgID        *string      `json:"org_id,omitempty"` // nullable, Phase 2.5 scaffold — global when NULL
	// OAuth status (never secrets): populated for oauth-backed providers.
	OAuthDefID     string  `json:"oauth_def_id,omitempty"`
	OAuthEmail     string  `json:"oauth_email,omitempty"`
	OAuthProject   string  `json:"oauth_project_id,omitempty"`
	OAuthExpires   *string `json:"oauth_expires_at,omitempty"`
	OAuthConnected bool    `json:"oauth_connected,omitempty"`
}

type GatewayKey struct {
	ID               string     `json:"id"`
	Name             string     `json:"name"`
	Prefix           string     `json:"prefix"`
	Hash             string     `json:"-"`
	Key              string     `json:"key,omitempty"` // only on creation
	LastUsedAt       *time.Time `json:"last_used_at"`
	CreatedAt        time.Time  `json:"created_at"`
	RevokedAt        *time.Time `json:"revoked_at"`
	RateLimitRPM     int        `json:"rate_limit_rpm"`
	RateLimitRPH     int        `json:"rate_limit_rph"`
	RateLimitRPD     int        `json:"rate_limit_rpd"`
	RateLimitTPM     int        `json:"rate_limit_tpm"`
	AllowedModels    []string   `json:"allowed_models,omitempty"`
	AllowedModelsRaw string     `json:"-"`                // raw JSON string from DB, not exposed
	OrgID            *string    `json:"org_id,omitempty"` // nullable, Phase 2.5 scaffold — global when NULL
	// CreatedBy is the dashboard user that created the key (keys:read_own
	// ownership anchor). NULL for legacy/unowned keys.
	CreatedBy *string `json:"created_by,omitempty"`

	// ExpiresAt gates authentication: Verify rejects keys past this time.
	// NULL = never expires.
	ExpiresAt *time.Time `json:"expires_at,omitempty"`
	// RotatedAt tracks the last secret rotation (drives the grace window;
	// shown as "rotated X ago" in the UI).
	RotatedAt *time.Time `json:"rotated_at,omitempty"`
	// IPAllowlist restricts gateway auth to matching client IPs (comma or
	// space separated IPs/CIDRs; empty = any IP).
	IPAllowlist string `json:"ip_allowlist,omitempty"`
	// MonthlyBudgetUSD caps calendar-month spend; 0 = unlimited.
	MonthlyBudgetUSD float64 `json:"monthly_usd_budget,omitempty"`
}

// Organization is Phase 2.5 pre-enterprise scaffold — nullable org_id on providers/keys.
// Existing rows have org_id=NULL meaning "global". Phase 3 will enforce isolation.
type Organization struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	CreatedAt time.Time `json:"created_at"`
}

// Membership links a user to an organization with a role (admin|member|readonly).
// Roles: admin, member, readonly — enforced by RequireRole middleware
type Membership struct {
	ID        string    `json:"id"`
	OrgID     string    `json:"org_id"`
	UserID    string    `json:"user_id"`
	Role      string    `json:"role"`
	CreatedAt time.Time `json:"created_at"`
}

type RequestLog struct {
	ID               string    `json:"id"`
	KeyPrefix        string    `json:"key_prefix"`
	ProviderID       string    `json:"provider_id"`
	Model            string    `json:"model"`
	Endpoint         string    `json:"endpoint"`
	Status           int       `json:"status"`
	LatencyMs        int64     `json:"latency_ms"`
	TTFTMs           int64     `json:"ttft_ms"`
	CreatedAt        time.Time `json:"created_at"`
	PromptTokens     int       `json:"prompt_tokens"`
	CompletionTokens int       `json:"completion_tokens"`
	TotalTokens      int       `json:"total_tokens"`
	CostUSD          float64   `json:"cost_usd"`
	IsStream         bool      `json:"is_stream"`
	Error            string    `json:"error,omitempty"`
	RequestBody      string    `json:"request_body,omitempty"`
	ResponseBody     string    `json:"response_body,omitempty"`
	FinishReason     string    `json:"finish_reason,omitempty"`
	FallbackChain    string    `json:"fallback_chain,omitempty"`
	CacheReadTokens  int       `json:"cache_read_tokens,omitempty"`
	CacheWriteTokens int       `json:"cache_write_tokens,omitempty"`
	ReasoningTokens  int       `json:"reasoning_tokens,omitempty"`
	// CacheHit marks rows served from the gateway's exact-match response
	// cache (X-Cache: HIT) — distinct from the upstream prompt-cache token
	// counters above.
	CacheHit bool `json:"cache_hit,omitempty"`
	// CacheStatus is the request's disposition against the response cache:
	// hit | miss | bypass. NULL on rows written before it existed.
	CacheStatus string `json:"cache_status,omitempty"`
}

// CatalogModel is enriched from models.dev
type CatalogModel struct {
	ID                    string    `json:"id"` // e.g. openai/gpt-5.5
	Provider              string    `json:"provider"`
	Name                  string    `json:"name"`
	Description           string    `json:"description"`
	Family                string    `json:"family"`
	ContextWindow         int       `json:"context_window"`
	MaxOutput             int       `json:"max_output"`
	InputCost             float64   `json:"input_cost"`  // per 1M
	OutputCost            float64   `json:"output_cost"` // per 1M
	CacheReadCost         float64   `json:"cache_read_cost"`
	CacheWriteCost        float64   `json:"cache_write_cost"`
	Reasoning             bool      `json:"reasoning"`
	ToolCall              bool      `json:"tool_call"`
	StructuredOutput      bool      `json:"structured_output"`
	Attachment            bool      `json:"attachment"`
	Modalities            string    `json:"modalities"` // json string
	OpenWeights           bool      `json:"open_weights"`
	KnowledgeCutoff       string    `json:"knowledge_cutoff"`
	UpdatedAt             time.Time `json:"updated_at"`
	ReasoningType         string    `json:"reasoning_type"`          // effort, toggle, none
	ReasoningLevels       string    `json:"reasoning_levels"`        // JSON array e.g. ["low","medium","high"]
	ReasoningOutputLimits string    `json:"reasoning_output_limits"` // JSON map e.g. {"low":10000,"high":50000}
}

type ModelAlias struct {
	Alias     string    `json:"alias"`
	Target    string    `json:"target"`
	CreatedAt time.Time `json:"created_at"`
}

type ProviderModel struct {
	ID                    string    `json:"id"`
	ProviderID            string    `json:"provider_id"`
	ProviderName          string    `json:"provider_name,omitempty"`
	ProviderType          string    `json:"provider_type,omitempty"`
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
	Source                string    `json:"source"`
	CreatedAt             time.Time `json:"created_at"`
	UpdatedAt             time.Time `json:"updated_at"`
	ReasoningType         string    `json:"reasoning_type"`
	ReasoningLevels       string    `json:"reasoning_levels"`
	ReasoningOutputLimits string    `json:"reasoning_output_limits"`
	AvgTTFTMs             float64   `json:"avg_ttft_ms,omitempty"`
	AvgTPS                float64   `json:"avg_tps,omitempty"`
	RequestCount          int       `json:"request_count,omitempty"`
}

// ExcludedModel is a provider model an operator removed. It stays out of
// auto-discovery until restored or re-added by hand. Snapshot is the row as
// it was when removed; legacy exclusions (written before the recycling bin)
// have an empty snapshot and restore as a plain discovered model.
type ExcludedModel struct {
	ID           string         `json:"id"`
	ProviderID   string         `json:"provider_id"`
	ProviderName string         `json:"provider_name,omitempty"`
	ModelID      string         `json:"model_id"`
	RemovedAt    time.Time      `json:"removed_at"`
	Snapshot     *ProviderModel `json:"snapshot,omitempty"`
}
