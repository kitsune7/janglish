// Command record captures audio from the default microphone, trims leading
// and trailing silence via a simple energy-based VAD, and writes the result as
// a 16 kHz mono FLAC file into data/manual/chris/. Filenames are assigned as
// s1.flac, s2.flac, ... with each run picking max+1 from the directory.
package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"

	"github.com/gen2brain/malgo"
	"github.com/kitsune7/janglish/internal/audio"
)

const outputDir = "data/manual/chris"

func main() {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		fmt.Fprintln(os.Stderr, "ffmpeg not found in PATH; install it with `brew install ffmpeg`")
		os.Exit(1)
	}

	outPath, err := nextFilename(outputDir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to pick output filename: %v\n", err)
		os.Exit(1)
	}

	pcm, err := captureUntilSilence()
	if err != nil {
		fmt.Fprintf(os.Stderr, "capture failed: %v\n", err)
		os.Exit(1)
	}

	if len(pcm) == 0 {
		fmt.Fprintln(os.Stderr, "no speech captured; nothing written")
		os.Exit(1)
	}

	if err := writeFLAC(pcm, outPath); err != nil {
		fmt.Fprintf(os.Stderr, "failed to encode FLAC: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Saved %s (%d samples, %.2fs)\n", outPath, len(pcm), float64(len(pcm))/float64(audio.SampleRate))
}

// nextFilename returns the path for the next sN.flac file in dir, creating
// dir if it does not already exist.
func nextFilename(dir string) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	re := regexp.MustCompile(`^s(\d+)\.flac$`)
	max := 0
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		m := re.FindStringSubmatch(e.Name())
		if m == nil {
			continue
		}
		n, err := strconv.Atoi(m[1])
		if err == nil && n > max {
			max = n
		}
	}
	return filepath.Join(dir, fmt.Sprintf("s%d.flac", max+1)), nil
}

// captureUntilSilence opens the default capture device at 16 kHz mono int16,
// runs the VAD until it reports end-of-speech, and returns the trimmed PCM.
func captureUntilSilence() ([]int16, error) {
	ctx, err := malgo.InitContext(nil, malgo.ContextConfig{}, nil)
	if err != nil {
		return nil, fmt.Errorf("init malgo context: %w", err)
	}
	defer func() {
		_ = ctx.Uninit()
		ctx.Free()
	}()

	cfg := malgo.DefaultDeviceConfig(malgo.Capture)
	cfg.Capture.Format = malgo.FormatS16
	cfg.Capture.Channels = 1
	cfg.SampleRate = audio.SampleRate
	cfg.PeriodSizeInFrames = audio.FrameSamples

	v := audio.NewVAD()
	done := make(chan struct{})
	var doneOnce bool
	var carry []int16

	onData := func(_, in []byte, frameCount uint32) {
		if doneOnce {
			return
		}
		samples := bytesToInt16(in[:int(frameCount)*2])
		carry = append(carry, samples...)
		for len(carry) >= audio.FrameSamples {
			frame := carry[:audio.FrameSamples]
			switch v.Feed(frame) {
			case audio.EventSpeechStart:
				fmt.Println("Listening...")
			case audio.EventSpeechEnd:
				fmt.Println("End of speech detected.")
				doneOnce = true
				close(done)
				return
			}
			carry = carry[audio.FrameSamples:]
		}
	}

	device, err := malgo.InitDevice(ctx.Context, cfg, malgo.DeviceCallbacks{Data: onData})
	if err != nil {
		return nil, fmt.Errorf("init capture device: %w", err)
	}
	if err := device.Start(); err != nil {
		device.Uninit()
		return nil, fmt.Errorf("start capture: %w", err)
	}

	fmt.Println("Waiting for speech... (speak now)")
	<-done
	device.Uninit()
	return v.Captured(), nil
}

// bytesToInt16 reinterprets a little-endian byte slice as []int16.
func bytesToInt16(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[i*2 : i*2+2]))
	}
	return out
}

// writeFLAC pipes the PCM samples through ffmpeg to produce a FLAC file.
func writeFLAC(pcm []int16, outPath string) error {
	cmd := exec.Command("ffmpeg",
		"-hide_banner", "-loglevel", "error", "-y",
		"-f", "s16le",
		"-ar", strconv.Itoa(audio.SampleRate),
		"-ac", "1",
		"-i", "-",
		"-c:a", "flac",
		outPath,
	)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	buf := make([]byte, len(pcm)*2)
	for i, s := range pcm {
		binary.LittleEndian.PutUint16(buf[i*2:], uint16(s))
	}
	if _, err := stdin.Write(buf); err != nil {
		_ = stdin.Close()
		_ = cmd.Wait()
		return err
	}
	if err := stdin.Close(); err != nil {
		return err
	}
	if err := cmd.Wait(); err != nil {
		if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
			return fmt.Errorf("ffmpeg exited %d", exitErr.ExitCode())
		}
		return err
	}
	return nil
}

