package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunAddWantedMovesBalancedSentences(t *testing.T) {
	tmp := t.TempDir()
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(oldWD); err != nil {
			t.Fatalf("restore cwd: %v", err)
		}
	})
	if err := os.Chdir(tmp); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir("data", 0o755); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(wantedCSV, []byte("Name|Sentence\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var mixed strings.Builder
	mixed.WriteString("Sentence|Gloss\n")
	for i := 1; i <= 10; i++ {
		fmt.Fprintf(&mixed, "This sentence has 日本語 word number %d.|English gloss %d\n", i, i)
	}
	for i := 1; i <= 10; i++ {
		fmt.Fprintf(&mixed, "今日は新しいstyle%dを試します。|Japanese gloss %d\n", i, i)
	}
	mixed.WriteString("One leftover sentence with 日本語.|Leftover gloss\n")
	if err := os.WriteFile(mixedCSV, []byte(mixed.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := runAddWanted([]string{"TestUser"}); err != nil {
		t.Fatal(err)
	}

	wantedRows, err := readPipeCSV(wantedCSV)
	if err != nil {
		t.Fatal(err)
	}
	if len(wantedRows) != 20 {
		t.Fatalf("wanted row count = %d, want 20", len(wantedRows))
	}
	for _, row := range wantedRows {
		if row[0] != "TestUser" {
			t.Fatalf("wanted row name = %q, want TestUser", row[0])
		}
	}

	mixedRows, err := readPipeCSV(mixedCSV)
	if err != nil {
		t.Fatal(err)
	}
	if len(mixedRows) != 1 {
		t.Fatalf("mixed row count = %d, want 1", len(mixedRows))
	}
	if got, want := mixedRows[0][0], "One leftover sentence with 日本語."; got != want {
		t.Fatalf("remaining mixed sentence = %q, want %q", got, want)
	}

	if _, err := os.Stat(filepath.Join("data", "mixed-sentences.csv.tmp")); !os.IsNotExist(err) {
		t.Fatalf("temporary mixed CSV still exists: %v", err)
	}
}
