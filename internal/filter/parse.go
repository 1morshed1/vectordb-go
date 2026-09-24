package filter

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Parse turns a filter DSL string into a Filter AST. An empty string means "no
// filter" and returns MatchAll().
//
// Grammar (lowest to highest precedence):
//
//	expr    := orExpr
//	orExpr  := andExpr ( "OR" andExpr )*
//	andExpr := unary  ( "AND" unary  )*
//	unary   := "NOT" unary | primary
//	primary := "(" expr ")" | condition
//	condition := IDENT op value
//	            | IDENT "in" "[" value ( "," value )* "]"
//	op      := "==" | "!=" | ">" | ">=" | "<" | "<="
//	value   := STRING | NUMBER | "true" | "false"
//
// Field names are bare identifiers. Strings use single or double quotes.
// Keywords (AND/OR/NOT/in/true/false) are case-insensitive.
//
// This is a hand-written recursive-descent parser — the same shape as the parser
// in a small language. It is worth reading precisely because it is small enough
// to hold in your head yet demonstrates lexing, precedence, and error handling.
func Parse(s string) (Filter, error) {
	if strings.TrimSpace(s) == "" {
		return MatchAll(), nil
	}
	toks, err := lex(s)
	if err != nil {
		return nil, err
	}
	p := &parser{toks: toks}
	f, err := p.parseExpr()
	if err != nil {
		return nil, err
	}
	if !p.atEnd() {
		return nil, fmt.Errorf("filter: unexpected token %q at position %d", p.cur().val, p.cur().pos)
	}
	return f, nil
}

// -----------------------------------------------------------------------------
// Lexer
// -----------------------------------------------------------------------------

type tokKind uint8

const (
	tokIdent tokKind = iota
	tokString
	tokNumber
	tokBool
	tokOp // == != > >= < <=
	tokAnd
	tokOr
	tokNot
	tokIn
	tokLParen
	tokRParen
	tokLBracket
	tokRBracket
	tokComma
	tokEOF
)

type token struct {
	kind tokKind
	val  string
	num  float64
	b    bool
	pos  int
}

func lex(s string) ([]token, error) {
	var toks []token
	runes := []rune(s)
	i := 0
	emit := func(k tokKind, v string, pos int) { toks = append(toks, token{kind: k, val: v, pos: pos}) }

	for i < len(runes) {
		c := runes[i]
		switch {
		case unicode.IsSpace(c):
			i++
		case c == '(':
			emit(tokLParen, "(", i)
			i++
		case c == ')':
			emit(tokRParen, ")", i)
			i++
		case c == '[':
			emit(tokLBracket, "[", i)
			i++
		case c == ']':
			emit(tokRBracket, "]", i)
			i++
		case c == ',':
			emit(tokComma, ",", i)
			i++
		case c == '\'' || c == '"':
			str, next, err := lexString(runes, i)
			if err != nil {
				return nil, err
			}
			toks = append(toks, token{kind: tokString, val: str, pos: i})
			i = next
		case c == '=' || c == '!' || c == '<' || c == '>':
			op, next, err := lexOperator(runes, i)
			if err != nil {
				return nil, err
			}
			emit(tokOp, op, i)
			i = next
		case c == '-' || c == '+' || unicode.IsDigit(c):
			num, raw, next, err := lexNumber(runes, i)
			if err != nil {
				return nil, err
			}
			toks = append(toks, token{kind: tokNumber, val: raw, num: num, pos: i})
			i = next
		case isIdentStart(c):
			word, next := lexIdent(runes, i)
			toks = append(toks, classifyWord(word, i))
			i = next
		default:
			return nil, fmt.Errorf("filter: unexpected character %q at position %d", string(c), i)
		}
	}
	toks = append(toks, token{kind: tokEOF, pos: len(runes)})
	return toks, nil
}

func lexString(runes []rune, start int) (string, int, error) {
	quote := runes[start]
	var sb strings.Builder
	i := start + 1
	for i < len(runes) {
		c := runes[i]
		if c == '\\' && i+1 < len(runes) {
			// Support simple escapes: \' \" \\ .
			sb.WriteRune(runes[i+1])
			i += 2
			continue
		}
		if c == quote {
			return sb.String(), i + 1, nil
		}
		sb.WriteRune(c)
		i++
	}
	return "", 0, fmt.Errorf("filter: unterminated string starting at position %d", start)
}

func lexOperator(runes []rune, start int) (string, int, error) {
	c := runes[start]
	// Two-char operators.
	if start+1 < len(runes) {
		two := string(runes[start : start+2])
		switch two {
		case "==", "!=", ">=", "<=":
			return two, start + 2, nil
		}
	}
	// Single-char < and > are valid; bare = and ! are not.
	switch c {
	case '<', '>':
		return string(c), start + 1, nil
	default:
		return "", 0, fmt.Errorf("filter: invalid operator starting with %q at position %d (did you mean ==?)", string(c), start)
	}
}

func lexNumber(runes []rune, start int) (float64, string, int, error) {
	i := start
	if runes[i] == '+' || runes[i] == '-' {
		i++
	}
	for i < len(runes) && (unicode.IsDigit(runes[i]) || runes[i] == '.' || runes[i] == 'e' || runes[i] == 'E' || runes[i] == '+' || runes[i] == '-') {
		// The e/+/- allowances let us lex scientific notation like 1.5e-3. We
		// stop as soon as strconv would reject, handled below.
		i++
	}
	raw := string(runes[start:i])
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, "", 0, fmt.Errorf("filter: invalid number %q at position %d", raw, start)
	}
	return f, raw, i, nil
}

func isIdentStart(c rune) bool { return unicode.IsLetter(c) || c == '_' }
func isIdentPart(c rune) bool {
	return unicode.IsLetter(c) || unicode.IsDigit(c) || c == '_' || c == '.'
}

func lexIdent(runes []rune, start int) (string, int) {
	i := start
	for i < len(runes) && isIdentPart(runes[i]) {
		i++
	}
	return string(runes[start:i]), i
}

// classifyWord turns a bare word into a keyword token or an identifier/bool.
func classifyWord(word string, pos int) token {
	switch strings.ToLower(word) {
	case "and":
		return token{kind: tokAnd, val: word, pos: pos}
	case "or":
		return token{kind: tokOr, val: word, pos: pos}
	case "not":
		return token{kind: tokNot, val: word, pos: pos}
	case "in":
		return token{kind: tokIn, val: word, pos: pos}
	case "true":
		return token{kind: tokBool, val: word, b: true, pos: pos}
	case "false":
		return token{kind: tokBool, val: word, b: false, pos: pos}
	default:
		return token{kind: tokIdent, val: word, pos: pos}
	}
}

// -----------------------------------------------------------------------------
// Parser
// -----------------------------------------------------------------------------

type parser struct {
	toks []token
	i    int
}

func (p *parser) cur() token  { return p.toks[p.i] }
func (p *parser) atEnd() bool { return p.cur().kind == tokEOF }
func (p *parser) advance() token {
	t := p.toks[p.i]
	if p.i < len(p.toks)-1 {
		p.i++
	}
	return t
}

func (p *parser) parseExpr() (Filter, error) { return p.parseOr() }

func (p *parser) parseOr() (Filter, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}
	if p.cur().kind != tokOr {
		return left, nil
	}
	parts := []Filter{left}
	for p.cur().kind == tokOr {
		p.advance()
		next, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		parts = append(parts, next)
	}
	return &Or{Parts: parts}, nil
}

func (p *parser) parseAnd() (Filter, error) {
	left, err := p.parseUnary()
	if err != nil {
		return nil, err
	}
	if p.cur().kind != tokAnd {
		return left, nil
	}
	parts := []Filter{left}
	for p.cur().kind == tokAnd {
		p.advance()
		next, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		parts = append(parts, next)
	}
	return &And{Parts: parts}, nil
}

func (p *parser) parseUnary() (Filter, error) {
	if p.cur().kind == tokNot {
		p.advance()
		inner, err := p.parseUnary()
		if err != nil {
			return nil, err
		}
		return &Not{Inner: inner}, nil
	}
	return p.parsePrimary()
}

func (p *parser) parsePrimary() (Filter, error) {
	if p.cur().kind == tokLParen {
		p.advance()
		inner, err := p.parseExpr()
		if err != nil {
			return nil, err
		}
		if p.cur().kind != tokRParen {
			return nil, fmt.Errorf("filter: expected ) at position %d", p.cur().pos)
		}
		p.advance()
		return inner, nil
	}
	return p.parseCondition()
}

func (p *parser) parseCondition() (Filter, error) {
	if p.cur().kind != tokIdent {
		return nil, fmt.Errorf("filter: expected a field name at position %d, got %q", p.cur().pos, p.cur().val)
	}
	field := p.advance().val

	// The "in" list form.
	if p.cur().kind == tokIn {
		p.advance()
		if p.cur().kind != tokLBracket {
			return nil, fmt.Errorf("filter: expected [ after 'in' at position %d", p.cur().pos)
		}
		p.advance()
		var list []Value
		for p.cur().kind != tokRBracket {
			v, err := p.parseValue()
			if err != nil {
				return nil, err
			}
			list = append(list, v)
			if p.cur().kind == tokComma {
				p.advance()
			} else if p.cur().kind != tokRBracket {
				return nil, fmt.Errorf("filter: expected , or ] at position %d", p.cur().pos)
			}
		}
		p.advance() // consume ]
		if len(list) == 0 {
			return nil, fmt.Errorf("filter: empty 'in' list for field %q", field)
		}
		return &Condition{Field: field, Op: OpIn, List: list}, nil
	}

	// The binary operator form.
	if p.cur().kind != tokOp {
		return nil, fmt.Errorf("filter: expected an operator after field %q at position %d", field, p.cur().pos)
	}
	op, err := parseOperator(p.advance().val)
	if err != nil {
		return nil, err
	}
	v, err := p.parseValue()
	if err != nil {
		return nil, err
	}
	return &Condition{Field: field, Op: op, Value: v}, nil
}

func (p *parser) parseValue() (Value, error) {
	t := p.cur()
	switch t.kind {
	case tokString:
		p.advance()
		return t.val, nil
	case tokNumber:
		p.advance()
		return t.num, nil
	case tokBool:
		p.advance()
		return t.b, nil
	default:
		return nil, fmt.Errorf("filter: expected a value at position %d, got %q", t.pos, t.val)
	}
}

func parseOperator(s string) (Operator, error) {
	switch s {
	case "==":
		return OpEq, nil
	case "!=":
		return OpNe, nil
	case ">":
		return OpGt, nil
	case ">=":
		return OpGte, nil
	case "<":
		return OpLt, nil
	case "<=":
		return OpLte, nil
	default:
		return 0, fmt.Errorf("filter: unknown operator %q", s)
	}
}
