package prom

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/opencost/opencost/core/pkg/source"
)

// recordingPromClient captures the PromQL of every query it is asked to run and
// returns an empty-but-valid Prometheus response, so query construction can be
// asserted without a Prometheus server.
type recordingPromClient struct {
	mu      sync.Mutex
	queries []string
}

func (c *recordingPromClient) URL(ep string, args map[string]string) *url.URL {
	return &url.URL{Scheme: "http", Host: "localhost:9090", Path: ep}
}

func (c *recordingPromClient) Do(_ context.Context, req *http.Request) (*http.Response, []byte, error) {
	if req != nil && req.URL != nil {
		if q := req.URL.Query().Get("query"); q != "" {
			c.mu.Lock()
			c.queries = append(c.queries, q)
			c.mu.Unlock()
		}
	}

	body := []byte(`{"status":"success","data":{"resultType":"vector","result":[]}}`)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	}, body, nil
}

func (c *recordingPromClient) recorded() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]string(nil), c.queries...)
}

func newRecordingQuerier() (*PrometheusMetricsQuerier, *recordingPromClient) {
	client := &recordingPromClient{}
	cfg := &OpenCostPrometheusConfig{ClusterLabel: "cluster_id"}
	return newPrometheusMetricsQuerier(cfg, client, NewContextFactory(client, cfg)), client
}

// TestQueryInferenceProcessingTime_MetricSelection pins the vLLM metric each
// processing-time query reads.
//
// Regression test for #4146: the output query used
// vllm:request_time_per_output_token_seconds_sum, whose _sum is a sum of
// per-request mean seconds-per-token rather than a DECODE duration, which
// inverted the compute-time cost split. The calculator tests take
// InputProcessingTime/OutputProcessingTime as given, so only an assertion at the
// query layer catches a reverted metric name.
func TestQueryInferenceProcessingTime_MetricSelection(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()

	tests := []struct {
		name     string
		run      func(*PrometheusMetricsQuerier) error
		want     string
		rejected string
	}{
		{
			name: "input uses prefill time",
			run: func(q *PrometheusMetricsQuerier) error {
				_, err := q.QueryInferenceInputProcessingTime(start, end).Await()
				return err
			},
			want: "vllm:request_prefill_time_seconds_sum",
		},
		{
			name: "output uses decode time, not time-per-output-token",
			run: func(q *PrometheusMetricsQuerier) error {
				_, err := q.QueryInferenceOutputProcessingTime(start, end).Await()
				return err
			},
			want:     "vllm:request_decode_time_seconds_sum",
			rejected: "vllm:request_time_per_output_token_seconds_sum",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			querier, client := newRecordingQuerier()

			if err := tc.run(querier); err != nil {
				t.Fatalf("query returned error: %v", err)
			}

			queries := client.recorded()

			// queryCounterDelta issues one query pinned to the start of the window and
			// one pinned to the end, per engine; both vLLM queries must read the metric.
			reads := 0
			for i, q := range queries {
				if strings.Contains(q, tc.want) {
					reads++
				}
				if tc.rejected != "" && strings.Contains(q, tc.rejected) {
					t.Errorf("boundary query %d reads %s, which is seconds-per-token rather than a duration: %s",
						i, tc.rejected, q)
				}
			}
			if reads != 2 {
				t.Errorf("expected 2 boundary queries reading %s, got %d: %v", tc.want, reads, queries)
			}
		})
	}
}

// valuePromClient answers each query with one series for model "m" in namespace
// "ns". End-of-window queries return the value configured for the metric they
// read; start-of-window queries (2m lookback) return 0. Metrics with no
// configured value return an empty vector, like an engine that is not deployed.
type valuePromClient struct {
	recordingPromClient
	endValues map[string]string
}

func (c *valuePromClient) Do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	// Record the query; the recording client's empty response is discarded.
	_, _, _ = c.recordingPromClient.Do(ctx, req)

	result := "[]"
	q := req.URL.Query().Get("query")
	for metric, value := range c.endValues {
		if !strings.Contains(q, metric+"[") && !strings.Contains(q, metric+"{") {
			continue
		}
		if strings.Contains(q, "[2m] @") {
			value = "0"
		}
		result = `[{"metric":{"model_name":"m","namespace":"ns"},"value":[1700000000,"` + value + `"]}]`
	}

	body := []byte(`{"status":"success","data":{"resultType":"vector","result":` + result + `}}`)
	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(body)),
	}, body, nil
}

func newValueQuerier(endValues map[string]string) *PrometheusMetricsQuerier {
	client := &valuePromClient{endValues: endValues}
	cfg := &OpenCostPrometheusConfig{ClusterLabel: "cluster_id"}
	return newPrometheusMetricsQuerier(cfg, client, NewContextFactory(client, cfg))
}

// TestQueryInferenceTokens_ReadsVLLMAndSGLang pins the token counters each token
// query reads: one per supported model server. Dropping one would silently zero
// the cost data for every model served by that engine (#4148). The SGLang
// selectors also pin the worker-role filter: in prefill/decode disaggregation
// both workers count every request, so prompt and cached tokens must skip
// decode workers and generation tokens must skip prefill workers.
func TestQueryInferenceTokens_ReadsVLLMAndSGLang(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()

	tests := []struct {
		name string
		run  func(*PrometheusMetricsQuerier) error
		want []string
	}{
		{
			name: "prompt tokens",
			run: func(q *PrometheusMetricsQuerier) error {
				_, err := q.QueryInferencePromptTokens(start, end).Await()
				return err
			},
			want: []string{"vllm:prompt_tokens_total", `sglang:prompt_tokens_total{engine_type!="decode"}`},
		},
		{
			name: "generation tokens",
			run: func(q *PrometheusMetricsQuerier) error {
				_, err := q.QueryInferenceGenerationTokens(start, end).Await()
				return err
			},
			want: []string{"vllm:generation_tokens_total", `sglang:generation_tokens_total{engine_type!="prefill"}`},
		},
		{
			name: "cached tokens",
			run: func(q *PrometheusMetricsQuerier) error {
				_, err := q.QueryInferenceCachedTokens(start, end).Await()
				return err
			},
			want: []string{"vllm:prefix_cache_hits_total", `sglang:cached_tokens_total{engine_type!="decode"}`},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			querier, client := newRecordingQuerier()

			if err := tc.run(querier); err != nil {
				t.Fatalf("query returned error: %v", err)
			}

			// Two boundary queries (start and end of window) per metric.
			queries := client.recorded()
			if len(queries) != 2*len(tc.want) {
				t.Fatalf("expected %d boundary queries, got %d: %v", 2*len(tc.want), len(queries), queries)
			}

			for _, metric := range tc.want {
				n := 0
				for _, q := range queries {
					if !strings.Contains(q, metric+"[") {
						continue
					}
					n++
					if !strings.Contains(q, "sum by (model_name, namespace)") {
						t.Errorf("query for %s is not grouped by (model_name, namespace): %s", metric, q)
					}
				}
				if n != 2 {
					t.Errorf("expected 2 boundary queries reading %s, got %d", metric, n)
				}
			}
		})
	}
}

// TestQueryInferenceTiming_SkipsModelsServedByBothEngines asserts each engine's
// timing is read only for models the other engine does not also serve, and the
// vLLM-only cache-config query skips SGLang-served models. A model served by
// both engines then falls back to the multiplier split and keeps SGLang's cache
// hits, instead of applying one engine's timing ratio or prefix-caching setting
// to the other engine's share.
func TestQueryInferenceTiming_SkipsModelsServedByBothEngines(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()
	const skipSGLang = `unless on (canonical_model_name, namespace) label_replace(sum by (model_name, namespace) (last_over_time(sglang:prompt_tokens_total[`
	const skipVLLM = `unless on (canonical_model_name, namespace) label_replace(sum by (model_name, namespace) (last_over_time(vllm:prompt_tokens_total[`

	querier, client := newRecordingQuerier()
	if _, err := querier.QueryInferenceInputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("input query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceOutputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("output query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceCacheConfig(end).Await(); err != nil {
		t.Fatalf("cache config query returned error: %v", err)
	}

	filtered := 0
	for _, q := range client.recorded() {
		isStart := strings.Contains(q, "[2m] @")
		switch {
		case strings.Contains(q, "group_left(enable_prefix_caching)") && strings.Contains(q, skipSGLang):
			// The cache-config join. Its diagnostic re-runs the join without the
			// filter to tell an expected empty result from a pod-label mismatch.
			filtered++
		case isStart:
			// Start-of-window boundary: deltas are keyed on the end query, so no filter needed.
		case strings.Contains(q, "vllm:request_") && strings.Contains(q, "_time_seconds_sum["):
			if !strings.Contains(q, skipSGLang) {
				t.Errorf("vLLM timing query does not skip SGLang-served models: %s", q)
			}
			filtered++
		case strings.Contains(q, "sglang:forward_execution_seconds_total{"):
			if !strings.Contains(q, skipVLLM) {
				t.Errorf("SGLang timing query does not skip vLLM-served models: %s", q)
			}
			filtered++
		}
	}
	if filtered != 5 {
		t.Errorf("expected 5 filtered queries (input and output end, per engine, plus cache config), got %d", filtered)
	}
}

// TestQueryInferenceTiming_ReadsSGLangForwardTime pins the SGLang processing-time
// selectors: prefill ("extend") on non-decode workers, and decode plus the
// speculative-decoding categories on non-prefill workers. "mixed" batches carry
// both phases and are left out of both sides.
func TestQueryInferenceTiming_ReadsSGLangForwardTime(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()

	querier, client := newRecordingQuerier()
	if _, err := querier.QueryInferenceInputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("input query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceOutputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("output query returned error: %v", err)
	}

	queries := strings.Join(client.recorded(), "\n")
	for _, want := range []string{
		`sglang:forward_execution_seconds_total{category="extend", engine_type!="decode"}[`,
		`sglang:forward_execution_seconds_total{category=~"decode|target_verify|draft_extend_v2|eagle_draft|eagle_draft_extend|frozen_kv_draft", engine_type!="prefill"}[`,
	} {
		if strings.Count(queries, want) != 2 {
			t.Errorf("expected start and end queries reading %s", want)
		}
	}
	if strings.Contains(queries, "mixed") {
		t.Errorf("mixed batches must not be counted as either phase")
	}
}

// TestQueryInferenceTiming_SumsSGLangPhases asserts SGLang forward time feeds the
// input and output processing times, which the calculator turns into a
// compute_time split.
func TestQueryInferenceTiming_SumsSGLangPhases(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()
	key := modelNamespaceKey("m", "ns")
	values := map[string]string{
		`sglang:forward_execution_seconds_total{category="extend", engine_type!="decode"}`:                                                                                "12",
		`sglang:forward_execution_seconds_total{category=~"decode|target_verify|draft_extend_v2|eagle_draft|eagle_draft_extend|frozen_kv_draft", engine_type!="prefill"}`: "48",
	}

	in, err := newValueQuerier(values).QueryInferenceInputProcessingTime(start, end).Await()
	if err != nil {
		t.Fatalf("input query returned error: %v", err)
	}
	out, err := newValueQuerier(values).QueryInferenceOutputProcessingTime(start, end).Await()
	if err != nil {
		t.Fatalf("output query returned error: %v", err)
	}

	sum := func(res []*source.InferenceProcessingTimeResult) float64 {
		total := 0.0
		for _, r := range res {
			total += r.Values[key]
		}
		return total
	}
	if got := sum(in); got != 12 {
		t.Errorf("input processing time = %v, want 12", got)
	}
	if got := sum(out); got != 48 {
		t.Errorf("output processing time = %v, want 48", got)
	}
}

// TestQueryInference_AppliesClusterFilter asserts every inference query applies
// the configured cluster filter to each metric it reads, so a Prometheus shared
// by several clusters does not add other clusters' tokens, or let another
// cluster's SGLang deployment suppress this cluster's vLLM timing.
func TestQueryInference_AppliesClusterFilter(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()
	const clusterFilter = `cluster_id="local"`

	client := &recordingPromClient{}
	cfg := &OpenCostPrometheusConfig{ClusterLabel: "cluster_id", ClusterFilter: clusterFilter}
	querier := newPrometheusMetricsQuerier(cfg, client, NewContextFactory(client, cfg))

	if _, err := querier.QueryInferencePromptTokens(start, end).Await(); err != nil {
		t.Fatalf("prompt query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceInputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("input query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceOutputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("output query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceCacheConfig(end).Await(); err != nil {
		t.Fatalf("cache config query returned error: %v", err)
	}

	want := []string{
		`sglang:forward_execution_seconds_total{category="extend", engine_type!="decode", ` + clusterFilter + `}[`,
		`engine_type!="prefill", ` + clusterFilter + `}[`,
		`last_over_time(vllm:prompt_tokens_total{` + clusterFilter + `}[`,
		`vllm:prompt_tokens_total{` + clusterFilter + `}[`,
		`sglang:prompt_tokens_total{engine_type!="decode", ` + clusterFilter + `}[`,
		`vllm:request_prefill_time_seconds_sum{` + clusterFilter + `}[`,
		`last_over_time(sglang:prompt_tokens_total{` + clusterFilter + `}[`,
		`vllm:cache_config_info{` + clusterFilter + `}`,
	}
	queries := strings.Join(client.recorded(), "\n")
	for _, w := range want {
		if !strings.Contains(queries, w) {
			t.Errorf("no query contains %s", w)
		}
	}
	for _, q := range client.recorded() {
		for _, metric := range []string{"vllm:prompt_tokens_total", "sglang:prompt_tokens_total", "vllm:cache_config_info", "sglang:forward_execution_seconds_total", "vllm:request_decode_time_seconds_sum"} {
			if strings.Contains(q, metric+"[") || strings.Contains(q, metric+")") {
				t.Errorf("query reads %s without the cluster filter: %s", metric, q)
			}
		}
	}
}

func TestWithClusterFilter(t *testing.T) {
	tests := []struct{ selector, filter, want string }{
		{"vllm:prompt_tokens_total", "", "vllm:prompt_tokens_total"},
		{"vllm:prompt_tokens_total", `cluster_id="a"`, `vllm:prompt_tokens_total{cluster_id="a"}`},
		{`sglang:prompt_tokens_total{engine_type!="decode"}`, `cluster_id="a"`, `sglang:prompt_tokens_total{engine_type!="decode", cluster_id="a"}`},
	}
	for _, tc := range tests {
		if got := withClusterFilter(tc.selector, tc.filter); got != tc.want {
			t.Errorf("withClusterFilter(%q, %q) = %q, want %q", tc.selector, tc.filter, got, tc.want)
		}
	}
}

// TestQueryInferenceTokens_SumsAcrossEngines asserts the per-engine deltas are
// added per (model_name, namespace), and that an engine with no series (not
// deployed) contributes nothing.
func TestQueryInferenceTokens_SumsAcrossEngines(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()
	key := modelNamespaceKey("m", "ns")

	tests := []struct {
		name      string
		endValues map[string]string
		want      float64
	}{
		{
			name:      "vLLM only",
			endValues: map[string]string{"vllm:prompt_tokens_total": "1000"},
			want:      1000,
		},
		{
			name:      "SGLang only",
			endValues: map[string]string{"sglang:prompt_tokens_total": "250"},
			want:      250,
		},
		{
			name: "both engines serve the model",
			endValues: map[string]string{
				"vllm:prompt_tokens_total":   "1000",
				"sglang:prompt_tokens_total": "250",
			},
			want: 1250,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := newValueQuerier(tc.endValues).QueryInferencePromptTokens(start, end).Await()
			if err != nil {
				t.Fatalf("query returned error: %v", err)
			}

			got := 0.0
			for _, r := range res {
				got += r.Values[key]
			}
			if got != tc.want {
				t.Errorf("prompt tokens for %s = %v, want %v", key, got, tc.want)
			}
		})
	}
}

// TestQueryInferenceProcessingTime_GroupsByModelAndNamespace asserts both
// processing-time queries aggregate by the labels the collector keys on. Dropping
// either label would silently collapse or discard per-model series.
func TestQueryInferenceProcessingTime_GroupsByModelAndNamespace(t *testing.T) {
	start := time.Now().Add(-time.Hour)
	end := time.Now()

	querier, client := newRecordingQuerier()

	if _, err := querier.QueryInferenceInputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("input query returned error: %v", err)
	}
	if _, err := querier.QueryInferenceOutputProcessingTime(start, end).Await(); err != nil {
		t.Fatalf("output query returned error: %v", err)
	}

	queries := client.recorded()
	if len(queries) != 8 {
		t.Fatalf("expected 8 boundary queries (2 phases x 2 engines x start/end), got %d", len(queries))
	}

	for i, q := range queries {
		if !strings.Contains(q, "sum by (model_name, namespace)") {
			t.Errorf("query %d is not grouped by (model_name, namespace): %s", i, q)
		}
	}
}
