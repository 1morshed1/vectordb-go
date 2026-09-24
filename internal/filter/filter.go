// Package filter implements metadata filtering for vector search: the
// "category == 'finance' AND year >= 2024" part of a query.
//
// It has two halves:
//   - An AST (Filter interface + Condition/And/Or/Not) with a Match method that
//     evaluates a filter against a single metadata map. This is what actually
//     decides eligibility.
//   - A small recursive-descent parser (parse.go) that turns a human-written
//     filter string into that AST, so the API can accept a readable expression.
//
// The strategy question — *when* to apply the filter relative to the vector
// search (pre vs post vs auto) — lives in the collection layer, which owns both
// the index and the metadata. See docs/learning/04.
package filter

import (
	"fmt"
	"strings"
)

// Value is a metadata value the filter can compare against: string, float64, or
// bool. This mirrors collection.Metadata's allowed value types.
type Value = any

// Filter is a boolean predicate over a metadata map.
type Filter interface {
	// Match reports whether the given metadata satisfies the filter.
	Match(md map[string]any) bool
	// String renders the filter back to its DSL form (useful for logging/tests).
	String() string
}

// Operator is a comparison operator in a Condition.
type Operator uint8

const (
	OpEq  Operator = iota // ==
	OpNe                  // !=
	OpGt                  // >
	OpGte                 // >=
	OpLt                  // <
	OpLte                 // <=
	OpIn                  // in [a, b, c]
)

func (o Operator) String() string {
	switch o {
	case OpEq:
		return "=="
	case OpNe:
		return "!="
	case OpGt:
		return ">"
	case OpGte:
		return ">="
	case OpLt:
		return "<"
	case OpLte:
		return "<="
	case OpIn:
		return "in"
	default:
		return "?"
	}
}

// Condition is a single leaf comparison: field OP value.
type Condition struct {
	Field string
	Op    Operator
	Value Value   // for OpIn, this is unused
	List  []Value // for OpIn, the set of allowed values
}

// Match evaluates the condition against one metadata map.
//
// A missing field never matches (returns false), including for != — we treat
// "the field isn't there" as "the condition can't be satisfied" rather than
// silently passing. This is the least-surprising behavior for filters like
// year >= 2024 on a document with no year.
func (c *Condition) Match(md map[string]any) bool {
	got, ok := md[c.Field]
	if !ok {
		return false
	}
	if c.Op == OpIn {
		for _, want := range c.List {
			if valuesEqual(got, want) {
				return true
			}
		}
		return false
	}
	return compare(got, c.Op, c.Value)
}

func (c *Condition) String() string {
	if c.Op == OpIn {
		parts := make([]string, len(c.List))
		for i, v := range c.List {
			parts[i] = formatValue(v)
		}
		return fmt.Sprintf("%s in [%s]", c.Field, strings.Join(parts, ", "))
	}
	return fmt.Sprintf("%s %s %s", c.Field, c.Op, formatValue(c.Value))
}

// And is satisfied only if every child is. An empty And matches everything
// (the identity of conjunction), which is convenient as a "no filter" value.
type And struct{ Parts []Filter }

func (a *And) Match(md map[string]any) bool {
	for _, p := range a.Parts {
		if !p.Match(md) {
			return false
		}
	}
	return true
}

func (a *And) String() string { return join(a.Parts, " AND ") }

// Or is satisfied if any child is. An empty Or matches nothing.
type Or struct{ Parts []Filter }

func (o *Or) Match(md map[string]any) bool {
	for _, p := range o.Parts {
		if p.Match(md) {
			return true
		}
	}
	return false
}

func (o *Or) String() string { return "(" + join(o.Parts, " OR ") + ")" }

// Not negates its child.
type Not struct{ Inner Filter }

func (n *Not) Match(md map[string]any) bool { return !n.Inner.Match(md) }
func (n *Not) String() string               { return "NOT " + n.Inner.String() }

func join(parts []Filter, sep string) string {
	s := make([]string, len(parts))
	for i, p := range parts {
		s[i] = p.String()
	}
	return strings.Join(s, sep)
}

// -----------------------------------------------------------------------------
// Value comparison. Only string, float64, and bool are supported; comparisons
// across different types are always false (a string is never < a number).
// -----------------------------------------------------------------------------

func valuesEqual(a, b any) bool {
	switch av := a.(type) {
	case string:
		bv, ok := b.(string)
		return ok && av == bv
	case float64:
		bv, ok := toFloat(b)
		return ok && av == bv
	case bool:
		bv, ok := b.(bool)
		return ok && av == bv
	default:
		// Numbers may arrive as int from Go callers; normalize via toFloat.
		if af, ok := toFloat(a); ok {
			if bf, ok2 := toFloat(b); ok2 {
				return af == bf
			}
		}
		return false
	}
}

// toFloat coerces the numeric types a caller might pass (int, int64, float32,
// float64) into a float64 for a single comparison path.
func toFloat(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}

// compare applies an ordered/equality operator to two values.
func compare(got any, op Operator, want any) bool {
	switch op {
	case OpEq:
		return valuesEqual(got, want)
	case OpNe:
		return !valuesEqual(got, want)
	}

	// Ordered comparisons only make sense for numbers.
	gf, ok1 := toFloat(got)
	wf, ok2 := toFloat(want)
	if !ok1 || !ok2 {
		return false
	}
	switch op {
	case OpGt:
		return gf > wf
	case OpGte:
		return gf >= wf
	case OpLt:
		return gf < wf
	case OpLte:
		return gf <= wf
	default:
		return false
	}
}

func formatValue(v any) string {
	switch x := v.(type) {
	case string:
		return fmt.Sprintf("%q", x)
	default:
		return fmt.Sprintf("%v", x)
	}
}

// MatchAll is the empty filter: it matches every metadata map. Represented as an
// empty And so callers can treat "no filter" uniformly with real filters.
func MatchAll() Filter { return &And{} }
