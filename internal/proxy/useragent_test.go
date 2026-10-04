package proxy

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"ai-gateway/internal/models"
)

// TestNewUpstreamRequestUserAgent pins the WAF-bypass behavior: the gateway
// must never relay Go's default User-Agent to a provider upstream. The
// Cloudflare WAF in front of inference-api.nousresearch.com (and others)
// answers "Go-http-client/1.1" with HTTP 403 error 1010, which silently broke
// model discovery, chat proxying, and health probes for those providers.
//
// Covered cases:
//   - no client UA (Go would inject its own)     -> gateway UA substituted
//   - client forwards Go's default verbatim       -> gateway UA substituted
//   - client sends a real UA (curl/httpx/browser) -> left intact
func TestNewUpstreamRequestUserAgent(t *testing.T) {
	h := &Handler{}
	ctx := context.Background()
	body := []byte(`{"model":"x","messages":[]}`)

	cases := []struct {
		name       string
		clientUA   string // "" means the header is absent entirely
		setHeader  bool
		wantUA     string
		wantExact  bool // true: want the exact gateway UA; false: want clientUA unchanged
		wantPrefix string
	}{
		{
			name:      "no client UA is replaced",
			setHeader: false,
			wantUA:    models.GatewayUserAgent,
			wantExact: true,
		},
		{
			name:      "go default UA is replaced",
			clientUA:  "Go-http-client/1.1",
			setHeader: true,
			wantUA:    models.GatewayUserAgent,
			wantExact: true,
		},
		{
			name:      "go default UA 2.0 is replaced",
			clientUA:  "Go-http-client/2.0",
			setHeader: true,
			wantUA:    models.GatewayUserAgent,
			wantExact: true,
		},
		{
			name:      "real curl UA is preserved",
			clientUA:  "curl/8.5.0",
			setHeader: true,
			wantUA:    "curl/8.5.0",
			wantExact: true,
		},
		{
			name:      "python httpx UA is preserved",
			clientUA:  "python-httpx/0.27.0",
			setHeader: true,
			wantUA:    "python-httpx/0.27.0",
			wantExact: true,
		},
		{
			name:      "browser UA is preserved",
			clientUA:  "Mozilla/5.0 (X11; Linux x86_64)",
			setHeader: true,
			wantUA:    "Mozilla/5.0 (X11; Linux x86_64)",
			wantExact: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, err := http.NewRequest("POST", "http://gateway.local/v1/chat/completions", nil)
			if err != nil {
				t.Fatalf("build client request: %v", err)
			}
			if tc.setHeader {
				r.Header.Set("User-Agent", tc.clientUA)
			}

			req, err := h.newUpstreamRequest(ctx, r, "https://inference-api.nousresearch.com/v1/chat/completions", "sk-test", body, false, false)
			if err != nil {
				t.Fatalf("newUpstreamRequest: %v", err)
			}

			got := req.Header.Get("User-Agent")
			if tc.wantExact {
				if got != tc.wantUA {
					t.Fatalf("User-Agent = %q, want %q", got, tc.wantUA)
				}
			}
			if strings.HasPrefix(got, "Go-http-client") {
				t.Fatalf("gateway relayed Go default User-Agent %q upstream (Cloudflare 403 bait)", got)
			}
			// Authorization must still be set to the decrypted provider key.
			if gotAuth := req.Header.Get("Authorization"); gotAuth != "Bearer sk-test" {
				t.Fatalf("Authorization = %q, want Bearer sk-test", gotAuth)
			}
		})
	}
}
