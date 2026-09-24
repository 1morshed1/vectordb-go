package filter

import "testing"

func md(pairs ...any) map[string]any {
	m := map[string]any{}
	for i := 0; i+1 < len(pairs); i += 2 {
		m[pairs[i].(string)] = pairs[i+1]
	}
	return m
}

func TestConditionComparisons(t *testing.T) {
	m := md("category", "finance", "year", float64(2024), "public", true)

	cases := []struct {
		expr string
		want bool
	}{
		{`category == "finance"`, true},
		{`category == "sports"`, false},
		{`category != "sports"`, true},
		{`year >= 2024`, true},
		{`year > 2024`, false},
		{`year <= 2024`, true},
		{`year < 2000`, false},
		{`public == true`, true},
		{`public == false`, false},
		{`missing == "x"`, false}, // missing field never matches
		{`missing != "x"`, false}, // even for !=
	}
	for _, c := range cases {
		f, err := Parse(c.expr)
		if err != nil {
			t.Fatalf("parse %q: %v", c.expr, err)
		}
		if got := f.Match(m); got != c.want {
			t.Errorf("%q matched %v, want %v", c.expr, got, c.want)
		}
	}
}

func TestLogicalCombinators(t *testing.T) {
	m := md("category", "finance", "year", float64(2024))
	cases := []struct {
		expr string
		want bool
	}{
		{`category == "finance" AND year >= 2024`, true},
		{`category == "finance" AND year >= 2025`, false},
		{`category == "sports" OR year >= 2024`, true},
		{`category == "sports" OR year >= 2025`, false},
		{`NOT category == "sports"`, true},
		{`NOT (category == "finance")`, false},
		{`(category == "finance" OR category == "sports") AND year == 2024`, true},
	}
	for _, c := range cases {
		f, err := Parse(c.expr)
		if err != nil {
			t.Fatalf("parse %q: %v", c.expr, err)
		}
		if got := f.Match(m); got != c.want {
			t.Errorf("%q matched %v, want %v", c.expr, got, c.want)
		}
	}
}

func TestInOperator(t *testing.T) {
	m := md("category", "finance")
	f, err := Parse(`category in ["finance", "tech", "health"]`)
	if err != nil {
		t.Fatal(err)
	}
	if !f.Match(m) {
		t.Error("expected finance to be in the list")
	}
	f2, _ := Parse(`category in ["tech", "health"]`)
	if f2.Match(m) {
		t.Error("finance should not match the list")
	}
}

func TestPrecedenceAndOverOr(t *testing.T) {
	// a OR b AND c should parse as a OR (b AND c).
	m := md("a", true, "b", false, "c", false)
	f, _ := Parse(`a == true OR b == true AND c == true`)
	if !f.Match(m) {
		t.Errorf("expected a OR (b AND c) to be true when a is true; got %s", f.String())
	}
}

func TestEmptyFilterMatchesAll(t *testing.T) {
	f, err := Parse("   ")
	if err != nil {
		t.Fatal(err)
	}
	if !f.Match(md("anything", "goes")) {
		t.Error("empty filter should match everything")
	}
}

func TestParseErrors(t *testing.T) {
	bad := []string{
		`category ==`,            // missing value
		`== "finance"`,           // missing field
		`category = "finance"`,   // single = is invalid
		`category in []`,         // empty in-list
		`(category == "finance"`, // unbalanced paren
		`category == "unterminated`,
	}
	for _, expr := range bad {
		if _, err := Parse(expr); err == nil {
			t.Errorf("expected parse error for %q", expr)
		}
	}
}

func TestNumberCoercionIntVsFloat(t *testing.T) {
	// A Go caller may store an int; comparisons should still work.
	m := md("year", 2024)
	f, _ := Parse(`year >= 2024`)
	if !f.Match(m) {
		t.Error("int metadata should compare equal to float literal")
	}
}
