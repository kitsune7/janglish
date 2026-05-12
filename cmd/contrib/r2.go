package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
)

// rcloneRemote returns the rclone remote prefix (e.g. "r2:janglish-recordings")
// from $JANGLISH_R2_REMOTE. Commands run by the contrib tool will all be
// relative to this prefix.
func rcloneRemote() (string, error) {
	remote := strings.TrimRight(os.Getenv(r2RemoteEnv), "/")
	if remote == "" {
		return "", fmt.Errorf("%s not set (expected something like r2:janglish-recordings; see docs/rclone-setup.md)", r2RemoteEnv)
	}
	if _, err := exec.LookPath("rclone"); err != nil {
		return "", errors.New("rclone not found in PATH; install it with `brew install rclone`")
	}
	return remote, nil
}

// r2Put writes body to <remote>/<key> via `rclone rcat`.
func r2Put(remote, key string, body []byte) error {
	cmd := exec.Command("rclone", "rcat", remote+"/"+key)
	cmd.Stdin = bytes.NewReader(body)
	cmd.Stdout = io.Discard
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// r2Copy recursively copies <remote>/<prefix> to the local destination via
// `rclone copy`. Missing remote prefixes are not an error — rclone just copies
// nothing.
func r2Copy(remote, prefix, dst string) error {
	cmd := exec.Command("rclone", "copy", remote+"/"+prefix, dst)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// r2Delete removes a single object at <remote>/<key>. Returns nil if the
// object does not exist (rclone reports this on stderr but exits non-zero, so
// we swallow the specific case).
func r2Delete(remote, key string) error {
	cmd := exec.Command("rclone", "deletefile", remote+"/"+key)
	var stderr bytes.Buffer
	cmd.Stdout = io.Discard
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if strings.Contains(stderr.String(), "object not found") {
			return nil
		}
		os.Stderr.Write(stderr.Bytes())
		return err
	}
	return nil
}

// r2Lsf returns file names (not full paths) under <remote>/<prefix> via
// `rclone lsf`. Directories are returned with a trailing slash.
func r2Lsf(remote, prefix string) ([]string, error) {
	cmd := exec.Command("rclone", "lsf", remote+"/"+prefix)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		if strings.Contains(stderr.String(), "directory not found") {
			return nil, nil
		}
		os.Stderr.Write(stderr.Bytes())
		return nil, err
	}
	var out []string
	for line := range strings.SplitSeq(stdout.String(), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	return out, nil
}
