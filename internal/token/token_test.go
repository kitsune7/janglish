package token

import (
	"strings"
	"testing"
	"time"
)

func TestSignVerifyRoundTrip(t *testing.T) {
	tok, err := Sign("s3cret", "alice", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	c, err := Verify("s3cret", tok)
	if err != nil {
		t.Fatal(err)
	}
	if c.Name != "alice" {
		t.Errorf("name = %q, want alice", c.Name)
	}
}

func TestVerifyRejectsBadSignature(t *testing.T) {
	tok, err := Sign("s3cret", "alice", time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify("wrong", tok); err == nil {
		t.Error("wanted error for bad secret")
	}
}

func TestVerifyRejectsExpired(t *testing.T) {
	tok, err := Sign("s3cret", "alice", -time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify("s3cret", tok); err == nil || !strings.Contains(err.Error(), "expired") {
		t.Errorf("wanted expired error, got %v", err)
	}
}

func TestVerifyRejectsMalformed(t *testing.T) {
	if _, err := Verify("s3cret", "not.a.token.really"); err == nil {
		t.Error("wanted malformed error")
	}
	if _, err := Verify("s3cret", "onlyonepart"); err == nil {
		t.Error("wanted malformed error")
	}
}
