package gcp

import (
	"os"
	"testing"

	"github.com/opencost/opencost/pkg/cloud/models"
)

func TestNormalizeTPULabel(t *testing.T) {
	cases := []struct {
		desc  string
		label string
		spot  bool
	}{
		{"TpuV5e running in Americas", "tpu-v5-lite-podslice", false},
		{"TpuV5e attached to Spot Preemptible VMs running in Americas", "tpu-v5-lite-podslice", true},
		{"TpuV5p running in Americas", "tpu-v5p-slice", false},
		{"TpuV6e running in Dallas", "tpu-v6e-slice", false},
		{"TPU7x running in Americas", "tpu7x", false},
		{"TPU7x attached to Spot Preemptible VMs running in Americas", "tpu7x", true},
		// Not a per-chip on-demand or spot price.
		{"Reserved V5e TPU in Iowa in Calendar Mode", "", false},
		{"Capacity Optimized TpuV6e running in London", "", false},
		{"DWS Defined Duration TPU7x running in Columbus", "", false},
		{"Commitment v1: TpuV5e running in Americas for 1 Year", "", false},
		{"Nvidia L4 GPU running in Americas", "", false},
	}
	for _, c := range cases {
		label, spot := NormalizeTPULabel(c.desc)
		if label != c.label || spot != c.spot {
			t.Errorf("NormalizeTPULabel(%q) = (%q, %v), want (%q, %v)", c.desc, label, spot, c.label, c.spot)
		}
	}
}

func tpuKey(instanceType, accelerator, region string, spot bool) *gcpKey {
	labels := map[string]string{
		"node.kubernetes.io/instance-type": instanceType,
		"topology.kubernetes.io/region":    region,
		GKE_TPU_TAG:                        accelerator,
	}
	if spot {
		labels[GKESpotLabel] = "true"
	}
	return &gcpKey{Labels: labels}
}

func TestTPUKey(t *testing.T) {
	key := tpuKey("ct5lp-hightpu-4t", "tpu-v5-lite-podslice", "us-central1", true)
	if got, want := key.Features(), "us-central1,ct5lphightpu,preemptible,gpu"; got != want {
		t.Errorf("Features() = %q, want %q", got, want)
	}
	if got, want := key.GPUType(), "tpu-v5-lite-podslice,preemptible"; got != want {
		t.Errorf("GPUType() = %q, want %q", got, want)
	}
}

// TestParsePageTPU parses real Compute Engine TPU SKUs (Cloud Billing Catalog,
// 8 Nov 2026): on-demand and spot v5e, v6e and TPU7x, plus reserved,
// commitment, capacity optimized and DWS SKUs that must not be used.
func TestParsePageTPU(t *testing.T) {
	f, err := os.Open("./test/tpu_skus.json")
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()

	keys := map[string]models.Key{}
	for _, k := range []*gcpKey{
		tpuKey("ct5lp-hightpu-1t", "tpu-v5-lite-podslice", "us-central1", false),
		tpuKey("ct5lp-hightpu-1t", "tpu-v5-lite-podslice", "us-central1", true),
		tpuKey("ct6e-standard-4t", "tpu-v6e-slice", "us-south1", false),
		tpuKey("tpu7x-standard-4t", "tpu7x", "us-central1", false),
		// No SKU in the page covers these: no price, rather than a wrong one.
		tpuKey("ct5lp-hightpu-1t", "tpu-v5-lite-podslice", "europe-west4", false),
		tpuKey("ct6e-standard-4t", "tpu-v6e-slice", "europe-west2", false),
	} {
		keys[k.Features()] = k
	}

	gcp := &GCP{}
	prices, _, err := gcp.parsePage(f, keys, map[string]models.PVKey{})
	if err != nil {
		t.Fatal(err)
	}

	want := map[string]struct{ sku, label, cost, usage string }{
		"us-central1,ct5lphightpu,ondemand,gpu":    {"8CCB-D501-DA04", "tpu-v5-lite-podslice", "1.2", "ondemand"},
		"us-central1,ct5lphightpu,preemptible,gpu": {"A9FF-6143-F24B", "tpu-v5-lite-podslice", "0.359329", "preemptible"},
		"us-south1,ct6estandard,ondemand,gpu":      {"2549-F208-72FA", "tpu-v6e-slice", "2.7", "ondemand"},
		"us-central1,tpu7xstandard,ondemand,gpu":   {"0F71-4391-B6CA", "tpu7x", "12", "ondemand"},
	}
	if len(prices) != len(want) {
		t.Errorf("got %d priced keys, want %d: %v", len(prices), len(want), keysOf(prices))
	}
	for k, w := range want {
		p, ok := prices[k]
		if !ok {
			t.Errorf("%s: no price", k)
			continue
		}
		n := p.Node
		if p.SKUID != w.sku || n.GPUName != w.label || n.GPUCost != w.cost || n.UsageType != w.usage {
			t.Errorf("%s: got SKU %s, %s at %s (%s); want SKU %s, %s at %s (%s)",
				k, p.SKUID, n.GPUName, n.GPUCost, n.UsageType, w.sku, w.label, w.cost, w.usage)
		}
		// The host is included in the chip price: zero, not empty, so the cost
		// model doesn't add default CPU and RAM prices.
		if n.VCPUCost != "0" || n.RAMCost != "0" || n.GPU != "1" {
			t.Errorf("%s: VCPUCost %q, RAMCost %q, GPU %q; want \"0\", \"0\", \"1\"", k, n.VCPUCost, n.RAMCost, n.GPU)
		}
		if !gcp.ValidPricingKeys[k] {
			t.Errorf("%s: not marked as a valid pricing key", k)
		}
	}
}

func keysOf(m map[string]*GCPPricing) []string {
	var ks []string
	for k := range m {
		ks = append(ks, k)
	}
	return ks
}
