// Package quantize implements scalar int8 quantization of float32 vectors.
//
// This is where ML intuition turns directly into an infrastructure win. A
// 1536-dimensional embedding stored as float32 is 1536 * 4 = 6144 bytes. The
// same vector as int8 is 1536 + a few bytes of scale = ~1540 bytes: a ~4x
// memory reduction. Because embeddings are trained to be robust and their
// components are smoothly distributed, rounding each dimension to one of 256
// levels costs very little search quality (recall). Memory is usually the
// binding constraint on how many vectors fit on a box, so 4x more vectors per
// dollar is a big deal.
//
// See docs/learning/05-quantization.md for the theory and the accuracy tests.
package quantize

import "math"

// QuantizedVector is a vector stored as int8 codes plus the scale needed to
// reconstruct approximate float32 values.
//
// We use *symmetric* per-vector quantization: one scale per vector, no offset.
// code_i in [-127, 127] maps to value_i ≈ code_i * Scale. Symmetric (centered
// on zero) suits embeddings, whose components are roughly zero-centered; it also
// makes the dot product — the operation we care about most — clean, because
// dot(a,b) ≈ ScaleA * ScaleB * Σ(codeA_i * codeB_i) with the sum done in
// integer arithmetic.
type QuantizedVector struct {
	Codes []int8
	Scale float32
}

// Quantize converts a float32 vector into int8 codes.
//
// Steps:
//  1. Find the largest absolute component, maxAbs. This is the value that must
//     map to the edge of the int8 range (±127) without clipping.
//  2. Scale = maxAbs / 127. Every component is divided by Scale and rounded to
//     the nearest integer, then clamped to [-127, 127] for safety.
//
// We reserve -128 (int8's true minimum) and use ±127 symmetrically so the
// mapping is exactly centered on zero. A zero vector gets Scale 0 and all-zero
// codes, which dequantizes back to zeros.
func Quantize(v []float32) QuantizedVector {
	var maxAbs float32
	for _, x := range v {
		a := x
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	codes := make([]int8, len(v))
	if maxAbs == 0 {
		return QuantizedVector{Codes: codes, Scale: 0}
	}
	scale := maxAbs / 127
	inv := 1 / scale
	for i, x := range v {
		q := int32(math.Round(float64(x * inv)))
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		codes[i] = int8(q)
	}
	return QuantizedVector{Codes: codes, Scale: scale}
}

// Dequantize reconstructs an approximate float32 vector from the codes.
func (q QuantizedVector) Dequantize() []float32 {
	out := make([]float32, len(q.Codes))
	for i, c := range q.Codes {
		out[i] = float32(c) * q.Scale
	}
	return out
}

// Dot computes the approximate dot product of two quantized vectors directly in
// the quantized domain. The elementwise products are accumulated as int32
// (fast, exact integer math), then scaled once at the end. This is the payoff
// of symmetric quantization: the hot inner loop is integer multiply-add, and
// modern CPUs run int8 dot products extremely fast (VNNI/dp4a instructions),
// though here we keep a plain, portable loop.
func Dot(a, b QuantizedVector) float32 {
	n := len(a.Codes)
	if len(b.Codes) < n {
		n = len(b.Codes)
	}
	var acc int32
	for i := 0; i < n; i++ {
		acc += int32(a.Codes[i]) * int32(b.Codes[i])
	}
	return float32(acc) * a.Scale * b.Scale
}

// MemoryBytes returns the on-heap byte cost of storing this quantized vector
// (codes + the scale). Used by the benchmark to report bytes-per-vector.
func (q QuantizedVector) MemoryBytes() int {
	return len(q.Codes) /* 1 byte each */ + 4 /* float32 scale */
}

// Float32Bytes returns the byte cost of the equivalent float32 vector, for the
// 4x comparison.
func Float32Bytes(dim int) int { return dim * 4 }

// ReconstructionError returns the mean absolute error between the original
// vector and its dequantized reconstruction. Lower is better; it quantifies how
// much precision we traded for the memory savings.
func ReconstructionError(original []float32, q QuantizedVector) float64 {
	rec := q.Dequantize()
	n := len(original)
	if len(rec) < n {
		n = len(rec)
	}
	if n == 0 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		sum += math.Abs(float64(original[i] - rec[i]))
	}
	return sum / float64(n)
}
