package audio

import (
	"math"
	"testing"
)

// makeFrame returns a frame filled with a sine wave at the given amplitude.
// amplitude=0 produces a silent frame.
func makeFrame(amplitude float64, phase *float64) []int16 {
	const freq = 440.0
	step := 2 * math.Pi * freq / float64(SampleRate)
	out := make([]int16, FrameSamples)
	for i := range out {
		out[i] = int16(amplitude * math.Sin(*phase))
		*phase += step
	}
	return out
}

func feedN(t *testing.T, v *VAD, n int, amp float64, phase *float64) []Event {
	t.Helper()
	events := make([]Event, 0, n)
	for range n {
		events = append(events, v.Feed(makeFrame(amp, phase)))
	}
	return events
}

func countEvent(events []Event, want Event) int {
	c := 0
	for _, e := range events {
		if e == want {
			c++
		}
	}
	return c
}

func TestVAD_SpeechStartAndEnd(t *testing.T) {
	v := NewVAD()
	var phase float64

	// 25 frames of silence — should stay in waiting, no events.
	silEvents := feedN(t, v, 25, 0, &phase)
	if countEvent(silEvents, EventSpeechStart) != 0 {
		t.Fatalf("unexpected speech-start during silence")
	}
	if v.State() != StateWaiting {
		t.Fatalf("expected StateWaiting, got %v", v.State())
	}

	// Loud frames — expect one SpeechStart after MinSpeechFrames.
	loudEvents := feedN(t, v, 40, 8000, &phase)
	if countEvent(loudEvents, EventSpeechStart) != 1 {
		t.Fatalf("expected exactly one SpeechStart, got %d", countEvent(loudEvents, EventSpeechStart))
	}
	if v.State() != StateSpeaking {
		t.Fatalf("expected StateSpeaking after loud frames, got %v", v.State())
	}

	// Long silence — should trigger SpeechEnd once MaxTrailingFrames is hit.
	endEvents := feedN(t, v, MaxTrailingFrames+5, 0, &phase)
	if countEvent(endEvents, EventSpeechEnd) != 1 {
		t.Fatalf("expected exactly one SpeechEnd, got %d", countEvent(endEvents, EventSpeechEnd))
	}
	if v.State() != StateDone {
		t.Fatalf("expected StateDone, got %v", v.State())
	}
}

func TestVAD_CapturedIncludesPreRoll(t *testing.T) {
	v := NewVAD()
	var phase float64

	// Prime the pre-roll with silence, then trigger speech.
	feedN(t, v, PreRollFrames+5, 0, &phase)
	feedN(t, v, MinSpeechFrames, 8000, &phase)

	// Pre-roll samples should have been flushed into captured.
	if len(v.Captured()) < PreRollFrames*FrameSamples {
		t.Fatalf("expected captured to include pre-roll (%d samples), got %d",
			PreRollFrames*FrameSamples, len(v.Captured()))
	}
}

func TestVAD_CapturedTrimsTrailingSilence(t *testing.T) {
	v := NewVAD()
	var phase float64

	feedN(t, v, MinSpeechFrames, 8000, &phase)           // speech start
	feedN(t, v, 10, 8000, &phase)                        // some spoken audio
	feedN(t, v, MaxTrailingFrames+2, 0, &phase)          // trigger speech end

	got := len(v.Captured())
	// Captured samples should retain only TailFrames of the trailing silence.
	// Upper bound: pre-roll + speaking (MinSpeechFrames+10) + TailFrames.
	maxExpected := (PreRollFrames + MinSpeechFrames + 10 + TailFrames + 1) * FrameSamples
	if got > maxExpected {
		t.Fatalf("trailing silence not trimmed: got %d samples, expected ≤ %d", got, maxExpected)
	}
	if got < (MinSpeechFrames+10)*FrameSamples {
		t.Fatalf("captured audio too short: %d samples", got)
	}
}

func TestFrameRMS(t *testing.T) {
	if r := FrameRMS(make([]int16, FrameSamples)); r != 0 {
		t.Fatalf("silent frame RMS should be 0, got %v", r)
	}
	loud := make([]int16, FrameSamples)
	for i := range loud {
		loud[i] = 10000
	}
	if r := FrameRMS(loud); r < 9000 || r > 11000 {
		t.Fatalf("constant-10000 RMS out of range: %v", r)
	}
}
