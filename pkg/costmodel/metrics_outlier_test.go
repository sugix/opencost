package costmodel

import "testing"

func TestIsCostOutlier(t *testing.T) {
	cases := []struct {
		name          string
		cost, average float64
		want          bool
	}{
		// A TPU node: the host is included in the chip price, so CPU and RAM cost 0.
		{"zero price, zero average", 0, 0, false},
		{"normal price", 0.0316, 0.0316, false},
		{"just under the cutoff", 29, 1, false},
		{"at the cutoff", 30, 1, false},
		{"spike", 31, 1, true},
		{"any price against a zero average", 0.0316, 0, true},
	}
	for _, c := range cases {
		if got := isCostOutlier(c.cost, c.average); got != c.want {
			t.Errorf("%s: isCostOutlier(%v, %v) = %v, want %v", c.name, c.cost, c.average, got, c.want)
		}
	}
}
