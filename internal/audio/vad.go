// Package audio contains audio-processing primitives shared across the
// janglish CLI programs. vad.go implements an energy-based voice-activity
// detector that streams 16-bit mono PCM frames.
package audio

import "math"

// Tuning constants. The VAD operates on fixed-size frames sampled at
// SampleRate. These values were chosen for a conversational microphone close
// to the speaker; tune SilenceThresholdRMS first if detection misfires.
const (
	SampleRate         = 16000
	FrameSamples       = 320 // 20 ms @ 16 kHz
	SilenceThresholdRMS = 500.0
	MinSpeechFrames    = 10  // ~200 ms of above-threshold audio to enter speaking
	MaxTrailingFrames  = 60  // ~1.2 s of below-threshold audio ends the clip
	PreRollFrames      = 50  // ~1 s of pre-speech audio retained on trigger
	TailFrames         = 8   // ~160 ms of silence kept at the end of the clip
)

// State represents the VAD state machine.
type State int

const (
	StateWaiting State = iota
	StateSpeaking
	StateTrailing
	StateDone
)

// Frame is a single fixed-size chunk of mono int16 PCM.
type Frame [FrameSamples]int16

// FrameRMS returns the root-mean-square energy of a frame.
func FrameRMS(f []int16) float64 {
	if len(f) == 0 {
		return 0
	}
	var sumSq float64
	for _, s := range f {
		v := float64(s)
		sumSq += v * v
	}
	return math.Sqrt(sumSq / float64(len(f)))
}

// Event describes a transition reported by VAD.Feed.
type Event int

const (
	EventNone Event = iota
	EventSpeechStart
	EventSpeechEnd
)

// VAD is a streaming energy-based voice-activity detector. Feed it one frame
// at a time; it maintains a pre-roll ring buffer so the first word's attack is
// preserved in the captured PCM returned by Captured.
type VAD struct {
	state State

	aboveRun int // consecutive above-threshold frames while waiting
	silRun   int // consecutive below-threshold frames while trailing

	preRoll     []int16 // ring buffer, length = PreRollFrames*FrameSamples
	preRollHead int     // next write index (wraps)
	preRollLen  int     // valid samples in ring (≤ cap)

	captured []int16 // PCM accumulated once speech started
}

// NewVAD returns a VAD ready to consume frames.
func NewVAD() *VAD {
	return &VAD{
		preRoll: make([]int16, PreRollFrames*FrameSamples),
	}
}

// State returns the current VAD state.
func (v *VAD) State() State { return v.state }

// Feed processes one frame of PCM and returns any state-transition event.
// The frame must be exactly FrameSamples long; shorter trailing frames should
// be zero-padded by the caller or discarded.
func (v *VAD) Feed(frame []int16) Event {
	if len(frame) != FrameSamples || v.state == StateDone {
		return EventNone
	}
	loud := FrameRMS(frame) >= SilenceThresholdRMS

	switch v.state {
	case StateWaiting:
		v.writePreRoll(frame)
		if loud {
			v.aboveRun++
			if v.aboveRun >= MinSpeechFrames {
				v.flushPreRoll()
				v.state = StateSpeaking
				return EventSpeechStart
			}
		} else {
			v.aboveRun = 0
		}

	case StateSpeaking:
		v.captured = append(v.captured, frame...)
		if !loud {
			v.state = StateTrailing
			v.silRun = 1
		}

	case StateTrailing:
		v.captured = append(v.captured, frame...)
		if loud {
			v.state = StateSpeaking
			v.silRun = 0
		} else {
			v.silRun++
			if v.silRun >= MaxTrailingFrames {
				v.state = StateDone
				return EventSpeechEnd
			}
		}
	}
	return EventNone
}

// Captured returns the PCM accumulated between speech start and speech end,
// with trailing silence trimmed to TailFrames.
func (v *VAD) Captured() []int16 {
	if len(v.captured) == 0 {
		return nil
	}
	// Trim trailing silence: we held MaxTrailingFrames of silence at the end;
	// keep only TailFrames of it.
	trim := (MaxTrailingFrames - TailFrames) * FrameSamples
	if trim < 0 || trim > len(v.captured) {
		return v.captured
	}
	return v.captured[:len(v.captured)-trim]
}

func (v *VAD) writePreRoll(frame []int16) {
	n := copy(v.preRoll[v.preRollHead:], frame)
	if n < len(frame) {
		copy(v.preRoll, frame[n:])
	}
	v.preRollHead = (v.preRollHead + len(frame)) % len(v.preRoll)
	v.preRollLen += len(frame)
	if v.preRollLen > len(v.preRoll) {
		v.preRollLen = len(v.preRoll)
	}
}

func (v *VAD) flushPreRoll() {
	if v.preRollLen == 0 {
		return
	}
	start := (v.preRollHead - v.preRollLen + len(v.preRoll)) % len(v.preRoll)
	if start+v.preRollLen <= len(v.preRoll) {
		v.captured = append(v.captured, v.preRoll[start:start+v.preRollLen]...)
	} else {
		v.captured = append(v.captured, v.preRoll[start:]...)
		v.captured = append(v.captured, v.preRoll[:v.preRollLen-(len(v.preRoll)-start)]...)
	}
}
