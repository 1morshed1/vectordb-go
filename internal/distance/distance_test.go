package distance

import (
	"math"
	"testing"
)

// approx reports whether two float32 values are equal within a small epsilon.
// Floating-point math is not associative, so exact equality is the wrong test.
func approx(a, b float32) bool {
	const eps = 1e-5
	return math.Abs(float64(a-b)) < eps
}

func TestParseMetric(t *testing.T) {
	cases := map[string]Metric{
		"":            Cosine,
		"cosine":      Cosine,
		"dot":         DotProduct,
		"inner_product": DotProduct,
		"l2":          Euclidean,
		"euclidean":   Euclidean,
	}
	for in, want := range cases {
		got, err := ParseMetric(in)
		if err != nil {
			t.Fatalf("ParseMetric(%q) error: %v", in, err)
		}
		if got != want {
			t.Errorf("ParseMetric(%q) = %v, want %v", in, got, want)
		}
	}
	if _, err := ParseMetric("nonsense"); err == nil {
		t.Error("ParseMetric(nonsense) should error")
	}
}

func TestDotProduct(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if got := dot(a, b); !approx(got, 32) {
		t.Errorf("dot = %v, want 32", got)
	}
	// Distance is the negated dot product.
	if got := Distance(DotProduct, a, b); !approx(got, -32) {
		t.Errorf("Distance(dot) = %v, want -32", got)
	}
}

func TestEuclidean(t *testing.T) {
	a := []float32{0, 0}
	b := []float32{3, 4}
	// classic 3-4-5 triangle
	if got := Distance(Euclidean, a, b); !approx(got, 5) {
		t.Errorf("euclidean = %v, want 5", got)
	}
}

func TestCosineIdenticalDirection(t *testing.T) {
	// Same direction, different magnitude => cosine distance 0.
	a := []float32{1, 0, 0}
	b := []float32{7, 0, 0}
	if got := Distance(Cosine, a, b); !approx(got, 0) {
		t.Errorf("cosine(same dir) = %v, want 0", got)
	}
}

func TestCosineOrthogonal(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	// 90 degrees => similarity 0 => distance 1.
	if got := Distance(Cosine, a, b); !approx(got, 1) {
		t.Errorf("cosine(orthogonal) = %v, want 1", got)
	}
}

func TestCosineOpposite(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{-1, 0}
	// opposite => similarity -1 => distance 2.
	if got := Distance(Cosine, a, b); !approx(got, 2) {
		t.Errorf("cosine(opposite) = %v, want 2", got)
	}
}

func TestZeroVectorIsMaximallyDistant(t *testing.T) {
	a := []float32{0, 0, 0}
	b := []float32{1, 2, 3}
	if got := Distance(Cosine, a, b); !approx(got, 2) {
		t.Errorf("cosine(zero) = %v, want 2 (max distance)", got)
	}
}

// TestNormalizeThenDotEqualsCosine proves the key optimization: after
// normalizing both vectors, the dot-product distance equals the cosine
// distance computed on the originals. This is why we normalize at insert time.
func TestNormalizeThenDotEqualsCosine(t *testing.T) {
	a := []float32{3, 1, 2, 5}
	b := []float32{1, 4, 1, 2}

	cosDist := Distance(Cosine, a, b)

	na := Normalize(append([]float32(nil), a...))
	nb := Normalize(append([]float32(nil), b...))
	// dot distance = -dot; cosine distance = 1 - sim = 1 - dot(normalized).
	// So cosineDist should equal 1 + Distance(DotProduct, na, nb).
	dotDist := Distance(DotProduct, na, nb)
	if !approx(cosDist, 1+dotDist) {
		t.Errorf("cosine %v != 1 + dotDist %v", cosDist, 1+dotDist)
	}
}

func TestNormalizeUnitLength(t *testing.T) {
	v := Normalize([]float32{3, 4})
	if got := Norm(v); !approx(got, 1) {
		t.Errorf("norm after normalize = %v, want 1", got)
	}
}

func TestNormalizeZeroVectorUnchanged(t *testing.T) {
	v := Normalize([]float32{0, 0, 0})
	for _, x := range v {
		if x != 0 {
			t.Errorf("zero vector changed: %v", v)
		}
	}
}
