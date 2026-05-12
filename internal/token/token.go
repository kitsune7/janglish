// Package token signs and verifies short contributor-auth tokens.
//
// Format: base64url(payload) + "." + base64url(hmacSHA256(secret, payload_b64))
// where payload is compact JSON {"name":"alice","exp":1730000000}. The Worker
// implements the same scheme in TypeScript.
package token

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

type Claims struct {
	Name string `json:"name"`
	Exp  int64  `json:"exp"`
}

// Sign returns a token for the given name that expires after ttl.
func Sign(secret, name string, ttl time.Duration) (string, error) {
	if secret == "" {
		return "", errors.New("empty secret")
	}
	if name == "" {
		return "", errors.New("empty name")
	}
	payload, err := json.Marshal(Claims{Name: name, Exp: time.Now().Add(ttl).Unix()})
	if err != nil {
		return "", err
	}
	payloadB64 := base64.RawURLEncoding.EncodeToString(payload)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(payloadB64))
	sig := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return payloadB64 + "." + sig, nil
}

// Verify parses and checks a token, returning its claims on success.
func Verify(secret, tok string) (*Claims, error) {
	parts := strings.Split(tok, ".")
	if len(parts) != 2 {
		return nil, errors.New("malformed token")
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(parts[0]))
	want := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(parts[1]), []byte(want)) {
		return nil, errors.New("bad signature")
	}
	raw, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return nil, fmt.Errorf("decode payload: %w", err)
	}
	var c Claims
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("parse payload: %w", err)
	}
	if time.Unix(c.Exp, 0).Before(time.Now()) {
		return nil, errors.New("token expired")
	}
	return &c, nil
}
