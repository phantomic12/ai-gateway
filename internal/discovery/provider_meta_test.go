package discovery

import (
	"testing"
)

// parseModelList must capture identity fields plus provider-declared detail
// from the same /models payload.
func TestParseModelListCapturesProviderMeta(t *testing.T) {
	body := []byte(`{"object":"list","data":[
		{"id":"m-plain","object":"model","owned_by":"acme"},
		{"id":"m-rich","object":"model","owned_by":"acme","display_name":"Rich Model",
		 "context_length":256000,"max_completion_tokens":8192,
		 "pricing":{"prompt":"0.0000002","completion":"0.0000008"},
		 "supported_parameters":["tools","reasoning"],
		 "architecture":{"modality":"text+image"}
		}
	]}`)
	out := parseModelList(body)
	if len(out) != 2 {
		t.Fatalf("got %d models, want 2", len(out))
	}
	plain, rich := out[0], out[1]
	if plain.ID != "m-plain" || plain.OwnedBy != "acme" || plain.Meta != nil {
		t.Fatalf("plain row: %+v", plain)
	}
	if rich.Meta == nil {
		t.Fatal("rich row must carry provider meta")
	}
	if rich.Meta.ContextWindow != 256000 || rich.Meta.MaxOutput != 8192 {
		t.Fatalf("meta windows: %+v", rich.Meta)
	}
	if rich.Meta.InputCost == nil {
		t.Fatal("input cost not declared")
	}
	if rich.Meta.OutputCost == nil {
		t.Fatal("output cost not declared")
	}
	t.Logf("input=%v output=%v", *rich.Meta.InputCost, *rich.Meta.OutputCost)
	if *rich.Meta.InputCost != 0.2 {
		t.Fatalf("input cost must be per-1M (got %v)", *rich.Meta.InputCost)
	}
	if *rich.Meta.OutputCost != 0.8 {
		t.Fatalf("output cost must be per-1M (got %v)", *rich.Meta.OutputCost)
	}
	if rich.Meta.ToolCall == nil || !*rich.Meta.ToolCall {
		t.Fatal("tools capability not detected")
	}
	if rich.Meta.Reasoning == nil || !*rich.Meta.Reasoning {
		t.Fatal("reasoning capability not detected")
	}
	if rich.Meta.Attachment == nil || !*rich.Meta.Attachment {
		t.Fatal("image_inputs -> attachment not detected")
	}
	if rich.Meta.Modalities == "" {
		t.Fatal("modalities not captured")
	}
	if rich.DisplayName != "Rich Model" {
		t.Fatalf("display_name: %q", rich.DisplayName)
	}
}

// Entries without id fall back to display_name; empty entries are skipped.
func TestParseModelListSkipsEmptyIDs(t *testing.T) {
	body := []byte(`{"object":"list","data":[{"id":"a"},{"display_name":"Named Only"},{"owned_by":"x"}]}`)
	out := parseModelList(body)
	if len(out) != 2 || out[1].ID != "Named Only" {
		t.Fatalf("got %+v", out)
	}
}

// Declared provider detail must override the catalog row per-field; declared
// $0.00 pricing must win over catalog pricing (pointer-vs-absent semantics).
func TestResolveEnrichmentProviderOverridesCatalog(t *testing.T) {
	s := &Service{} // nil catalogStore: catalog side produces zeros
	zero := 0.0
	m := rawModel{ID: "x", Meta: &providerMeta{
		ContextWindow: 7777,
		InputCost:     &zero, OutputCost: &zero,
	}}
	e := s.resolveEnrichment(m)
	if e.source != "provider" {
		t.Fatalf("source=%q want provider", e.source)
	}
	if e.ctx != 7777 {
		t.Fatalf("ctx=%d", e.ctx)
	}
	if e.inputCost != 0 || e.outputCost != 0 {
		t.Fatalf("declared free pricing must stay 0, got %v/%v", e.inputCost, e.outputCost)
	}
}

// No declared detail: catalog path unchanged.
func TestResolveEnrichmentNoMetaFallsBack(t *testing.T) {
	s := &Service{}
	e := s.resolveEnrichment(rawModel{ID: "x"})
	if e.source != "discovered" {
		t.Fatalf("source=%q want discovered (no catalog, no meta)", e.source)
	}
}

// parseLeakedLimit extracts the real cap from reseller 400 bodies.
func TestParseLeakedLimit(t *testing.T) {
	cases := []struct {
		body string
		want int
	}{
		{`{"error":{"message":"max_completion_tokens=999999 cannot be greater than max_model_len=max_total_tokens=262144"}}`, 262144},
		{`max_model_len: 131072`, 131072},
		{`context length = 65536`, 65536},
		{`{"error":"unauthorized"}`, 0},
		{`random text`, 0},
		{`max_model_len=99999999999`, 0}, // absurd — rejected
	}
	for _, c := range cases {
		if got := parseLeakedLimit(c.body); got != c.want {
			t.Errorf("parseLeakedLimit(%q) = %d, want %d", c.body, got, c.want)
		}
	}
}
