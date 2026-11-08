package gcp

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/opencost/opencost/core/pkg/log"
	"github.com/opencost/opencost/pkg/cloud/models"
)

// GKE_TPU_TAG is the node label GKE sets on Cloud TPU nodes, e.g.
// "tpu-v5-lite-podslice". See
// https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus
const GKE_TPU_TAG = "cloud.google.com/gke-tpu-accelerator"

// tpuSKURegex matches the Compute Engine SKUs that carry the per-chip on-demand
// or spot price of a TPU, e.g. "TpuV5e running in Americas" and "TpuV5e attached
// to Spot Preemptible VMs running in Americas". It is anchored so that reserved
// ("Reserved V5e TPU in Iowa in Calendar Mode"), capacity optimized, DWS and
// commitment SKUs never match.
var tpuSKURegex = regexp.MustCompile(`^(TpuV5e|TpuV5p|TpuV6e|TPU7x)( attached to Spot Preemptible VMs)? running in `)

// tpuSKUToLabel maps the TPU name in a SKU description to the value of the
// GKE_TPU_TAG node label. TPU v4 and older are billed by the Cloud TPU service,
// not Compute Engine, so they are not listed.
var tpuSKUToLabel = map[string]string{
	"TpuV5e": "tpu-v5-lite-podslice",
	"TpuV5p": "tpu-v5p-slice",
	"TpuV6e": "tpu-v6e-slice",
	"TPU7x":  "tpu7x",
}

// NormalizeTPULabel returns the GKE_TPU_TAG value priced by a Compute Engine
// SKU, and whether the SKU is the spot price. It returns "" for SKUs that are
// not a per-chip on-demand or spot price.
func NormalizeTPULabel(desc string) (label string, spot bool) {
	m := tpuSKURegex.FindStringSubmatch(desc)
	if m == nil {
		return "", false
	}
	return tpuSKUToLabel[m[1]], m[2] != ""
}

// addTPUPricing prices the TPU nodes in inputKeys from one Compute Engine SKU
// whose resource group is TPU.
//
// A TPU machine is billed per chip with its host VM included: the catalog has
// no vCPU or RAM SKU for TPU machine types. So the node gets the chip price as
// its GPU price, and an explicit zero vCPU and RAM price. Left empty, those
// would make the cost model fall back to the default CPU and RAM prices and
// charge for the host twice. The chip count comes from the node's
// google.com/tpu capacity.
func (gcp *GCP) addTPUPricing(product *GCPPricing, usageType string, inputKeys map[string]models.Key, out map[string]*GCPPricing) error {
	label, spot := NormalizeTPULabel(product.Description)
	if label == "" {
		return nil
	}
	// Commitment SKUs have their own usage types; a spot description must
	// come with the Preemptible usage type and an on-demand one with OnDemand.
	if (usageType != "ondemand" && usageType != "preemptible") || spot != (usageType == "preemptible") {
		return nil
	}
	if len(product.PricingInfo) == 0 || product.PricingInfo[0].PricingExpression == nil {
		return nil
	}
	rates := product.PricingInfo[0].PricingExpression.TieredRates
	if len(rates) == 0 || rates[len(rates)-1].UnitPrice == nil {
		return nil
	}
	unitPrice := rates[len(rates)-1].UnitPrice
	units, err := strconv.Atoi(unitPrice.Units)
	if err != nil {
		return fmt.Errorf("error parsing base unit price for tpu: %w", err)
	}
	// as per https://cloud.google.com/billing/v1/how-tos/catalog-api
	// the hourly price is the whole currency price + the fractional currency price
	hourlyPrice := unitPrice.Nanos*math.Pow10(-9) + float64(units)
	if hourlyPrice == 0 {
		return nil
	}

	if gcp.ValidPricingKeys == nil {
		gcp.ValidPricingKeys = make(map[string]bool)
	}
	for k, key := range inputKeys {
		if key.GPUType() != label+","+usageType {
			continue
		}
		if !slices.Contains(product.ServiceRegions, strings.Split(k, ",")[0]) {
			continue
		}
		features := key.Features()
		log.Debugf("GCP Billing API: matched TPU to node: %s: %s", features, product.Description)
		gcp.ValidPricingKeys[features] = true
		pricing := *product
		pricing.Node = &models.Node{
			VCPUCost:  "0",
			RAMCost:   "0",
			GPU:       "1",
			GPUName:   label,
			GPUCost:   strconv.FormatFloat(hourlyPrice, 'f', -1, 64),
			UsageType: usageType,
		}
		out[features] = &pricing
	}
	return nil
}
