// Command contrib manages remote contributor recordings.
//
// Subcommands:
//
//	contrib pull
//	    Transcodes WAV uploads under work/uploads/manual/<name>/<id>.wav into
//	    16 kHz mono FLAC under data/manual/<name>/<id>.flac, deletes the WAVs,
//	    and appends matching rows to data/data-pairs.csv. Idempotent.
//
//	contrib gen-assignment <name> <id> [<id>...]
//	    Looks up sentence IDs in data/wanted-sentences.csv and writes
//	    work/uploads/assignments/<name>.json with the assigned sentences. In
//	    Phase 1 this is a local file; Phase 2 will upload it to R2.
//
//	contrib gen-token <name> [ttl]
//	    Prints a signed HMAC token for the contributor. ttl defaults to 168h
//	    (7 days); accepts any duration Go's time.ParseDuration understands.
//	    Requires JANGLISH_HMAC_SECRET in env or .env.
package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/kitsune7/janglish/internal/token"
)

const (
	wantedCSV       = "data/wanted-sentences.csv"
	pairsCSV        = "data/data-pairs.csv"
	uploadsDir      = "work/uploads/manual"
	assignmentsDir  = "work/uploads/assignments"
	manualOutputDir = "data/manual"
	placeholderHost = "https://example.github.io/janglish-record"
	r2RemoteEnv     = "JANGLISH_R2_REMOTE" // e.g. "r2:janglish-recordings"
	siteBaseEnv     = "JANGLISH_SITE_BASE" // e.g. "https://chris.github.io/janglish"
	defaultTTL      = 7 * 24 * time.Hour
)

func main() {
	if err := token.LoadDotEnv(".env"); err != nil {
		fmt.Fprintf(os.Stderr, "warning: load .env: %v\n", err)
	}
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "pull":
		if err := runPull(); err != nil {
			fmt.Fprintf(os.Stderr, "pull: %v\n", err)
			os.Exit(1)
		}
	case "gen-assignment":
		if err := runGenAssignment(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "gen-assignment: %v\n", err)
			os.Exit(1)
		}
	case "gen-token":
		if err := runGenToken(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "gen-token: %v\n", err)
			os.Exit(1)
		}
	case "list-progress":
		if err := runListProgress(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "list-progress: %v\n", err)
			os.Exit(1)
		}
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  contrib pull")
	fmt.Fprintln(os.Stderr, "  contrib gen-assignment <name> <id> [<id>...]")
	fmt.Fprintln(os.Stderr, "  contrib gen-token <name> [ttl]")
	fmt.Fprintln(os.Stderr, "  contrib list-progress [name]")
}

// runGenToken prints a signed HMAC token for <name>. The matching secret
// belongs in JANGLISH_HMAC_SECRET (env or .env at repo root). ttl defaults
// to 168h (7 days).
func runGenToken(args []string) error {
	if len(args) < 1 {
		return errors.New("expected: <name> [ttl]")
	}
	name := args[0]
	if !validName(name) {
		return fmt.Errorf("invalid name %q: must match [a-z0-9][a-z0-9-]*", name)
	}
	ttl := defaultTTL
	if len(args) >= 2 {
		d, err := time.ParseDuration(args[1])
		if err != nil {
			return fmt.Errorf("parse ttl: %w", err)
		}
		ttl = d
	}
	secret := os.Getenv("JANGLISH_HMAC_SECRET")
	if secret == "" {
		return errors.New("JANGLISH_HMAC_SECRET not set (add it to .env)")
	}
	tok, err := token.Sign(secret, name, ttl)
	if err != nil {
		return err
	}
	fmt.Println(tok)
	return nil
}

// runPull downloads new WAV uploads from R2 into work/uploads/manual/, transcodes
// each into 16 kHz mono FLAC under data/manual/<name>/, deletes the WAV from R2
// (so the staging area drains end-to-end), removes the local WAV copy, and
// appends new pairs to data/data-pairs.csv. Re-runs are safe: rclone copy is a
// no-op on unchanged objects, already-transcoded WAVs are gone from R2, and
// CSV duplicates are skipped.
func runPull() error {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return errors.New("ffmpeg not found in PATH; install it with `brew install ffmpeg`")
	}

	wanted, err := loadWanted()
	if err != nil {
		return err
	}

	remote, err := rcloneRemote()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(uploadsDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", uploadsDir, err)
	}
	if err := r2Copy(remote, "manual/", uploadsDir); err != nil {
		return fmt.Errorf("rclone copy from %s/manual/: %w", remote, err)
	}

	wavs, err := findWAVs(uploadsDir)
	if err != nil {
		return err
	}
	if len(wavs) == 0 {
		fmt.Println("no uploads found under", uploadsDir)
		return nil
	}

	type newPair struct{ path, text string }
	var added []newPair
	var skipped []string

	for _, wav := range wavs {
		rel, err := filepath.Rel(uploadsDir, wav)
		if err != nil {
			return fmt.Errorf("relpath %s: %w", wav, err)
		}
		parts := strings.Split(filepath.ToSlash(rel), "/")
		if len(parts) != 2 {
			skipped = append(skipped, fmt.Sprintf("%s (expected <name>/<id>.wav layout)", wav))
			continue
		}
		name := parts[0]
		id := strings.TrimSuffix(parts[1], filepath.Ext(parts[1]))

		text, ok := wanted[id]
		if !ok {
			skipped = append(skipped, fmt.Sprintf("%s (id %q not in %s)", wav, id, wantedCSV))
			continue
		}

		outDir := filepath.Join(manualOutputDir, name)
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return fmt.Errorf("mkdir %s: %w", outDir, err)
		}
		outPath := filepath.Join(outDir, id+".flac")
		if err := transcodeWAVToFLAC(wav, outPath); err != nil {
			return fmt.Errorf("transcode %s: %w", wav, err)
		}

		r2Key := "manual/" + name + "/" + id + ".wav"
		if err := r2Delete(remote, r2Key); err != nil {
			return fmt.Errorf("r2 delete %s: %w", r2Key, err)
		}
		if err := os.Remove(wav); err != nil {
			return fmt.Errorf("remove %s: %w", wav, err)
		}
		fmt.Printf("transcoded %s -> %s\n", wav, outPath)

		pairPath := filepath.ToSlash(filepath.Join("manual", name, id+".flac"))
		added = append(added, newPair{path: pairPath, text: text})
	}

	if len(added) > 0 {
		paths := make([]string, len(added))
		texts := make([]string, len(added))
		for i, p := range added {
			paths[i] = p.path
			texts[i] = p.text
		}
		appended, err := appendPairs(paths, texts)
		if err != nil {
			return err
		}
		if appended > 0 {
			fmt.Printf("appended %d new row(s) to %s\n", appended, pairsCSV)
		}
	}

	for _, s := range skipped {
		fmt.Fprintln(os.Stderr, "skipped:", s)
	}
	return nil
}

// runGenAssignment writes a JSON assignment file for a contributor, uploads it
// to R2, signs a contributor token, and prints the share URL. The local copy
// under work/uploads/assignments/<name>.json is kept for inspection.
func runGenAssignment(args []string) error {
	if len(args) < 2 {
		return errors.New("expected: <name> <id> [<id>...]")
	}
	name := args[0]
	ids := args[1:]

	if !validName(name) {
		return fmt.Errorf("invalid name %q: must match [a-z0-9][a-z0-9-]*", name)
	}

	wanted, err := loadWanted()
	if err != nil {
		return err
	}

	type sentence struct {
		ID   string `json:"id"`
		Text string `json:"text"`
	}
	sentences := make([]sentence, 0, len(ids))
	for _, id := range ids {
		text, ok := wanted[id]
		if !ok {
			return fmt.Errorf("id %q not found in %s", id, wantedCSV)
		}
		sentences = append(sentences, sentence{ID: id, Text: text})
	}

	if err := os.MkdirAll(assignmentsDir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", assignmentsDir, err)
	}
	outPath := filepath.Join(assignmentsDir, name+".json")
	payload := struct {
		Name      string     `json:"name"`
		Sentences []sentence `json:"sentences"`
	}{Name: name, Sentences: sentences}

	buf, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	buf = append(buf, '\n')
	if err := os.WriteFile(outPath, buf, 0o644); err != nil {
		return fmt.Errorf("write %s: %w", outPath, err)
	}
	fmt.Printf("wrote %s (%d sentence(s))\n", outPath, len(sentences))

	remote, err := rcloneRemote()
	if err != nil {
		return err
	}
	r2Key := "assignments/" + name + ".json"
	if err := r2Put(remote, r2Key, buf); err != nil {
		return fmt.Errorf("upload %s: %w", r2Key, err)
	}
	fmt.Printf("uploaded %s/%s\n", remote, r2Key)

	secret := os.Getenv("JANGLISH_HMAC_SECRET")
	if secret == "" {
		return errors.New("JANGLISH_HMAC_SECRET not set (add it to .env)")
	}
	tok, err := token.Sign(secret, name, defaultTTL)
	if err != nil {
		return fmt.Errorf("sign token: %w", err)
	}

	base := strings.TrimRight(os.Getenv(siteBaseEnv), "/")
	if base == "" {
		base = placeholderHost
	}
	fmt.Printf("share: %s/?t=%s\n", base, tok)
	return nil
}

// nameRE matches the same shape the Worker enforces on token claims and R2
// keys: lowercase ASCII alnum plus dashes, not leading with a dash.
var nameRE = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

func validName(s string) bool { return nameRE.MatchString(s) }

// runListProgress prints per-contributor progress: how many sentences were
// assigned, how many WAVs are staged in R2 awaiting pull, and how many FLACs
// have landed in data/manual/. With no args it lists every contributor with
// an assignment in R2; with a name it narrows to that contributor.
func runListProgress(args []string) error {
	remote, err := rcloneRemote()
	if err != nil {
		return err
	}

	assignmentFiles, err := r2Lsf(remote, "assignments/")
	if err != nil {
		return err
	}

	names := make([]string, 0)
	for _, f := range assignmentFiles {
		if !strings.HasSuffix(f, ".json") {
			continue
		}
		names = append(names, strings.TrimSuffix(f, ".json"))
	}
	sort.Strings(names)

	if len(args) == 1 {
		want := args[0]
		filtered := names[:0]
		for _, n := range names {
			if n == want {
				filtered = append(filtered, n)
			}
		}
		if len(filtered) == 0 {
			return fmt.Errorf("no assignment for %q in %s/assignments/", want, remote)
		}
		names = filtered
	}

	if len(names) == 0 {
		fmt.Println("no assignments found in", remote+"/assignments/")
		return nil
	}

	fmt.Printf("%-20s %10s %10s %10s\n", "contributor", "assigned", "staged", "done")
	for _, name := range names {
		assigned, err := countAssigned(remote, name)
		if err != nil {
			return err
		}
		staged, err := r2Lsf(remote, "manual/"+name+"/")
		if err != nil {
			return err
		}
		done := countLocalFLACs(name)
		fmt.Printf("%-20s %10d %10d %10d\n", name, assigned, len(staged), done)
	}
	return nil
}

// countAssigned fetches an assignment JSON from R2 and returns its sentence
// count. Returns 0 if the object is missing.
func countAssigned(remote, name string) (int, error) {
	cmd := exec.Command("rclone", "cat", remote+"/assignments/"+name+".json")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if strings.Contains(stderr.String(), "object not found") {
			return 0, nil
		}
		os.Stderr.Write(stderr.Bytes())
		return 0, err
	}
	var payload struct {
		Sentences []struct{} `json:"sentences"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		return 0, fmt.Errorf("parse assignment %s: %w", name, err)
	}
	return len(payload.Sentences), nil
}

// countLocalFLACs returns the number of *.flac files under data/manual/<name>/.
func countLocalFLACs(name string) int {
	entries, err := os.ReadDir(filepath.Join(manualOutputDir, name))
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if !e.IsDir() && strings.EqualFold(filepath.Ext(e.Name()), ".flac") {
			n++
		}
	}
	return n
}

// loadWanted reads data/wanted-sentences.csv into an id->text map. The file is
// pipe-delimited CSV with a header row.
func loadWanted() (map[string]string, error) {
	rows, err := readPipeCSV(wantedCSV)
	if err != nil {
		return nil, err
	}
	out := make(map[string]string, len(rows))
	for _, row := range rows {
		id := strings.TrimSpace(row[0])
		if id == "" {
			continue
		}
		if _, dup := out[id]; dup {
			return nil, fmt.Errorf("%s: duplicate id %q", wantedCSV, id)
		}
		out[id] = row[1]
	}
	return out, nil
}

// findWAVs returns every *.wav file under root, sorted for deterministic
// processing order. Returns an empty slice (not an error) if root does not
// exist yet.
func findWAVs(root string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return fs.SkipAll
			}
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Ext(path), ".wav") {
			out = append(out, path)
		}
		return nil
	})
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, err
	}
	sort.Strings(out)
	return out, nil
}

// transcodeWAVToFLAC runs ffmpeg to convert the input WAV into 16 kHz mono
// signed-16-bit FLAC, matching the format produced by `just record`.
func transcodeWAVToFLAC(in, out string) error {
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error", "-y",
		"-i", in,
		"-ar", "16000",
		"-ac", "1",
		"-sample_fmt", "s16",
		"-c:a", "flac",
		out,
	)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// appendPairs appends new rows to data/data-pairs.csv, skipping any paths
// already present. Opens the file in append mode so existing rows are never
// rewritten (preserves their exact formatting). Returns the number of rows
// actually added.
func appendPairs(paths, texts []string) (int, error) {
	existing, err := readPipeCSV(pairsCSV)
	if err != nil {
		return 0, err
	}
	seen := make(map[string]struct{}, len(existing))
	for _, row := range existing {
		seen[row[0]] = struct{}{}
	}

	toAdd := make([][]string, 0, len(paths))
	for i, p := range paths {
		if _, ok := seen[p]; ok {
			continue
		}
		toAdd = append(toAdd, []string{p, texts[i]})
		seen[p] = struct{}{}
	}
	if len(toAdd) == 0 {
		return 0, nil
	}

	f, err := os.OpenFile(pairsCSV, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o644)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	w := csv.NewWriter(f)
	w.Comma = '|'
	if err := w.WriteAll(toAdd); err != nil {
		return 0, err
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return 0, err
	}
	return len(toAdd), nil
}

// readPipeCSV reads a pipe-delimited CSV and returns the data rows (minus
// header). Tolerates bare double-quotes via LazyQuotes so sentences like
// `Tell him I say \"やだ\".` parse. Returns empty slice (not an error) if the
// file does not exist.
func readPipeCSV(path string) ([][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.Comma = '|'
	r.FieldsPerRecord = 2
	r.LazyQuotes = true

	records, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	if len(records) == 0 {
		return nil, nil
	}
	return records[1:], nil
}
