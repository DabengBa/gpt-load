package gateway

import (
	"bytes"
	"errors"
	"math/rand"
	"os"
	"strconv"
	"testing"
)

func TestResponsesSSEFramingNormalizerEventDataGapFixture(t *testing.T) {
	input, err := os.ReadFile("testdata/responses-event-data-gap.sse")
	if err != nil {
		t.Fatal(err)
	}
	want := bytes.Join([][]byte{
		[]byte("event: response.function_call_arguments.delta\n: keep-alive\n"),
		[]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"x\"}\n\n"),
		[]byte(": keep-alive\n\n"),
	}, nil)
	for _, chunkSize := range []int{1, 2, 7, len(input)} {
		t.Run(chunkSizeName(chunkSize), func(t *testing.T) {
			got, err := normalizeResponsesSSEFramingForTest(input, chunkSize)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("normalized fixture = %q, want %q", got, want)
			}
			if bytes.Contains(got, []byte("event: response.function_call_arguments.delta\n: keep-alive\n\n")) {
				t.Fatal("normalized fixture still contains an event-only block")
			}
		})
	}
}

func TestResponsesSSEFramingNormalizerPreservesLineEndingsAndChunks(t *testing.T) {
	for _, lineEnding := range []string{"\n", "\r\n", "\r"} {
		t.Run(lineEndingName(lineEnding), func(t *testing.T) {
			eventBlock := "event: response.output_text.delta" + lineEnding + ": keep-alive" + lineEnding + lineEnding
			dataBlock := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}" + lineEnding + lineEnding
			input := []byte(eventBlock + dataBlock)
			want := []byte("event: response.output_text.delta" + lineEnding + ": keep-alive" + lineEnding + "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}" + lineEnding + lineEnding)

			for _, chunkSize := range []int{1, 3, len(input)} {
				got, err := normalizeResponsesSSEFramingForTest(input, chunkSize)
				if err != nil {
					t.Fatalf("chunk size %d: %v", chunkSize, err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("chunk size %d: got %q, want %q", chunkSize, got, want)
				}
			}
		})
	}
}

func TestResponsesSSEFramingNormalizerAcceptsMultipleDataLines(t *testing.T) {
	input := []byte("event: response.output_text.delta\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\n" +
		"data: \"ok\"}\n\n")
	got, err := normalizeResponsesSSEFramingForTest(input, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\n" +
		"data: \"ok\"}\n\n")
	if !bytes.Equal(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestResponsesSSEFramingNormalizerBoundsOversizedSingleChunk(t *testing.T) {
	const limit = 128
	input := bytes.Repeat([]byte{'x'}, limit*1024)
	normalizer := newResponsesSSEFramingNormalizer(limit)
	if _, err := normalizer.push(input); !errors.Is(err, errSSEEventTooLarge) {
		t.Fatalf("push() error = %v, want %v", err, errSSEEventTooLarge)
	}
	if normalizer.peakPendingBytes > limit+1 {
		t.Fatalf("peak pending bytes = %d, want <= %d", normalizer.peakPendingBytes, limit+1)
	}
	if len(normalizer.pending) > limit+1 || cap(normalizer.pending) > limit+1 {
		t.Fatalf("failed normalizer retained pending len/cap = %d/%d, want <= %d", len(normalizer.pending), cap(normalizer.pending), limit+1)
	}
}

func TestResponsesSSEFramingNormalizerProcessesManyBlocksInLargeChunk(t *testing.T) {
	const limit = 64
	block := []byte(": keep-alive\n\n")
	input := bytes.Repeat(block, 128)
	normalizer := newResponsesSSEFramingNormalizer(limit)
	got, err := normalizer.push(input)
	if err != nil {
		t.Fatal(err)
	}
	final, err := normalizer.finish()
	if err != nil {
		t.Fatal(err)
	}
	got = append(got, final...)
	if !bytes.Equal(got, input) {
		t.Fatalf("large chunk output differs: got %d bytes, want %d", len(got), len(input))
	}
	if normalizer.peakPendingBytes > limit+1 {
		t.Fatalf("peak pending bytes = %d, want <= %d", normalizer.peakPendingBytes, limit+1)
	}
}

func TestResponsesSSEFramingNormalizerRejectsNonUTF8RecoveryComments(t *testing.T) {
	const eventName = "response.output_text.delta"
	validData := []byte("data: {\"type\":\"response.output_text.delta\"}\n\n")
	dataComment := append([]byte("event: "+eventName+"\n\n: "), 0xff)
	dataComment = append(dataComment, []byte("\ndata: {\"type\":\""+eventName+"\"}\n\n")...)
	tests := []struct {
		name  string
		input []byte
	}{
		{
			name: "event comment",
			input: append(append([]byte("event: "+eventName+"\n: "), 0xff),
				append([]byte("\n\n"), validData...)...),
		},
		{
			name:  "data comment",
			input: dataComment,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeResponsesSSEFramingForTest(test.input, 1)
			if !errors.Is(err, errResponsesSSEFraming) {
				t.Fatalf("error = %v, want %v", err, errResponsesSSEFraming)
			}
			if got != nil {
				t.Fatalf("invalid UTF-8 input returned output %q", got)
			}
		})
	}
}

func TestResponsesSSEFramingNormalizerAcceptsSplitCRLFDelimiter(t *testing.T) {
	input := []byte("event: response.output_text.delta\r\n\r\ndata: {\"type\":\"response.output_text.delta\"}\r\n\r\n")
	normalizer := newResponsesSSEFramingNormalizer(maxSSEEventBytes)
	var got bytes.Buffer
	for _, chunk := range splitBytes(input, 1) {
		output, err := normalizer.push(chunk)
		if err != nil {
			t.Fatal(err)
		}
		got.Write(output)
	}
	output, err := normalizer.finish()
	if err != nil {
		t.Fatal(err)
	}
	got.Write(output)
	want := []byte("event: response.output_text.delta\r\ndata: {\"type\":\"response.output_text.delta\"}\r\n\r\n")
	if !bytes.Equal(got.Bytes(), want) {
		t.Fatalf("got %q, want %q", got.Bytes(), want)
	}
}

func TestResponsesSSEFramingNormalizerRejectsPartialEOF(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{name: "event without data", input: "event: response.output_text.delta\n\n"},
		{name: "data without delimiter", input: "event: response.output_text.delta\n\ndata: {\"type\":\"response.output_text.delta\"}\n"},
		{name: "event line without block delimiter", input: "event: response.output_text.delta\n"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeResponsesSSEFramingForTest([]byte(test.input), 1)
			if !errors.Is(err, errSSEEventIncomplete) {
				t.Fatalf("error = %v, want %v", err, errSSEEventIncomplete)
			}
			if got != nil {
				t.Fatalf("partial input returned output %q", got)
			}
		})
	}
}

func TestResponsesSSEFramingNormalizerRejectsAmbiguousRecovery(t *testing.T) {
	const eventName = "response.output_text.delta"
	validData := `data: {"type":"response.output_text.delta"}`
	tests := []struct {
		name  string
		input string
	}{
		{name: "mismatched type", input: "event: " + eventName + "\n\ndata: {\"type\":\"response.completed\"}\n\n"},
		{name: "missing type", input: "event: " + eventName + "\n\ndata: {\"delta\":\"x\"}\n\n"},
		{name: "duplicate type", input: "event: " + eventName + "\n\ndata: {\"type\":\"response.output_text.delta\",\"type\":\"response.output_text.delta\"}\n\n"},
		{name: "non object", input: "event: " + eventName + "\n\ndata: []\n\n"},
		{name: "invalid UTF-8", input: "event: " + eventName + "\n\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\xff}\n\n"},
		{name: "empty data", input: "event: " + eventName + "\n\ndata:\n\n"},
		{name: "extra field", input: "event: " + eventName + "\n\nid: request-1\ndata: {\"type\":\"response.output_text.delta\"}\n\n"},
		{name: "non adjacent block", input: "event: " + eventName + "\n\n: keep-alive\n\n" + validData + "\n\n"},
		{name: "done payload", input: "event: " + eventName + "\n\ndata: [DONE]\n\n"},
		{name: "duplicate event", input: "event: " + eventName + "\nevent: " + eventName + "\n\n" + validData + "\n\n"},
		{name: "empty event", input: "event:\n\n" + validData + "\n\n"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeResponsesSSEFramingForTest([]byte(test.input), 1)
			if !errors.Is(err, errResponsesSSEFraming) {
				t.Fatalf("error = %v, want %v", err, errResponsesSSEFraming)
			}
			if got != nil {
				t.Fatalf("ambiguous input returned output %q", got)
			}
		})
	}
}

func TestResponsesSSEFramingNormalizerPreservesIndependentBlocks(t *testing.T) {
	input := []byte(": keep-alive\n\ndata: {\"type\":\"response.output_text.delta\"}\n\nevent: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\"}\n\nid: request-1\n\n")
	got, err := normalizeResponsesSSEFramingForTest(input, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, input) {
		t.Fatalf("independent blocks changed: got %q, want %q", got, input)
	}
}

func TestResponsesSSEFramingNormalizerRejectsMergedEventAboveLimit(t *testing.T) {
	input := []byte("event: response.output_text.delta\n\ndata: {\"type\":\"response.output_text.delta\"}\n\n")
	normalizer := newResponsesSSEFramingNormalizer(len(input) - 2)
	if _, err := normalizer.push(input); !errors.Is(err, errSSEEventTooLarge) {
		t.Fatalf("push() error = %v, want %v", err, errSSEEventTooLarge)
	}
	if _, err := normalizer.finish(); err == nil {
		t.Fatal("finish() unexpectedly succeeded after oversized merge")
	}
}

// TestResponsesSSEFramingNormalizerResumesScanLinearly is the B3 regression
// proof. It asserts, deterministically (no wall clock), two invariants on the
// full in-contract input class, including CR-runs, consecutive empty blocks and
// multi-block streams that force discard():
//
//  1. The absolute scan position (bytes discarded + relative cursor) never
//     moves backwards before the first error, so no byte is ever re-scanned.
//     The relative scanOffset itself legitimately resets on discard().
//  2. Total delimiter probes are O(N): the scan examines each fed byte at most
//     once, and the only re-examinations are deferred trailing '\r' bytes. Each
//     non-empty chunk is bracketed by a pre-append and a post-append consume,
//     and a consume can end in at most one '\r' deferral, so the budget is
//     N + 2*len(chunks), plus one for a trailing '\r' re-examined by finish().
//     Note a '\r'\n' pair costs one base probe (the '\r' iteration covers the
//     '\n'), so the base term stays below N.
//
// A rescan-from-zero implementation fails invariant 2 by orders of magnitude.
func TestResponsesSSEFramingNormalizerResumesScanLinearly(t *testing.T) {
	longLine := append([]byte("data: "), bytes.Repeat([]byte("x"), 16*1024)...)
	crlfLines := bytes.Repeat([]byte("data: x\r\n"), 2048)
	trailingCR := append(bytes.Repeat([]byte("data: x\r\n"), 2048), []byte("data: y\r")...)
	crRun := bytes.Repeat([]byte("\r"), 4096)
	crTerminatedBlocks := bytes.Repeat([]byte(": keep-alive\r\r"), 1024)
	emptyBlocks := bytes.Repeat([]byte("\n\n"), 2048)
	multiBlocks := bytes.Repeat([]byte(": keep-alive\n\n"), 1024)
	inputs := []struct {
		name  string
		input []byte
	}{
		{name: "single long line", input: longLine},
		{name: "crlf lines", input: crlfLines},
		{name: "trailing cr", input: trailingCR},
		{name: "cr run", input: crRun},
		{name: "cr terminated blocks", input: crTerminatedBlocks},
		{name: "consecutive empty blocks", input: emptyBlocks},
		{name: "many keep-alive blocks", input: multiBlocks},
	}
	chunkings := []struct {
		name  string
		chunk func([]byte) [][]byte
	}{
		{name: "one byte", chunk: chunkedBySize(1)},
		{name: "split at cr", chunk: chunkedSplitAtCarriageReturn},
		{name: "fixed random", chunk: chunkedRandomly(1, 9, 20260916)},
		{name: "large chunk", chunk: chunkedBySize(1 << 20)},
	}
	for _, input := range inputs {
		for _, chunking := range chunkings {
			t.Run(input.name+"/"+chunking.name, func(t *testing.T) {
				normalizer := newResponsesSSEFramingNormalizer(maxSSEEventBytes)
				chunks := chunking.chunk(input.input)
				ingested := 0
				previousAbsolute := 0
				for _, chunk := range chunks {
					if _, err := normalizer.push(chunk); err != nil {
						t.Fatalf("push() error = %v", err)
					}
					// These inputs never approach maxEventBytes, so push() ingests every
					// fed byte and discarded + len(pending) == ingested holds exactly.
					ingested += len(chunk)
					absolute := ingested - len(normalizer.pending) + normalizer.scanOffset
					if absolute < previousAbsolute {
						t.Fatalf("absolute scan position regressed from %d to %d", previousAbsolute, absolute)
					}
					previousAbsolute = absolute
				}
				// finish() may re-examine one trailing '\r' left deferred at EOF.
				_, _ = normalizer.finish()
				bound := len(input.input) + 2*len(chunks) + 1
				t.Logf("input=%d chunks=%d probes=%d bound=%d", len(input.input), len(chunks), normalizer.scanProbes, bound)
				if normalizer.scanProbes > bound {
					t.Fatalf("scan probes = %d, want <= %d (input %d bytes in %d chunks)",
						normalizer.scanProbes, bound, len(input.input), len(chunks))
				}
			})
		}
	}
}

// TestResponsesSSEFramingNormalizerFailureResetsScanCursor verifies the
// terminal-failure reset separately from the linear-scan proof: once a framing,
// size or EOF failure makes the normalizer failed, its incremental cursor and
// pending buffer are cleared so no stale scan state is retained or resumed.
func TestResponsesSSEFramingNormalizerFailureResetsScanCursor(t *testing.T) {
	t.Run("oversized pending", func(t *testing.T) {
		const limit = 64
		normalizer := newResponsesSSEFramingNormalizer(limit)
		if _, err := normalizer.push(bytes.Repeat([]byte("x"), limit+2)); !errors.Is(err, errSSEEventTooLarge) {
			t.Fatalf("push() error = %v, want %v", err, errSSEEventTooLarge)
		}
		assertResponsesSSEScanCursorReset(t, normalizer)
	})
	t.Run("partial eof", func(t *testing.T) {
		normalizer := newResponsesSSEFramingNormalizer(maxSSEEventBytes)
		if _, err := normalizer.push([]byte("data: {\"type\":\"response.completed\"}")); err != nil {
			t.Fatalf("push() error = %v", err)
		}
		if normalizer.scanOffset == 0 {
			t.Fatal("scan cursor did not advance before the failure")
		}
		if _, err := normalizer.finish(); !errors.Is(err, errSSEEventIncomplete) {
			t.Fatalf("finish() error = %v, want %v", err, errSSEEventIncomplete)
		}
		assertResponsesSSEScanCursorReset(t, normalizer)
	})
	t.Run("framing rejection", func(t *testing.T) {
		normalizer := newResponsesSSEFramingNormalizer(maxSSEEventBytes)
		input := []byte("event: response.output_text.delta\n\ndata: {\"type\":\"response.completed\"}\n\n")
		if _, err := normalizer.push(input); !errors.Is(err, errResponsesSSEFraming) {
			t.Fatalf("push() error = %v, want %v", err, errResponsesSSEFraming)
		}
		assertResponsesSSEScanCursorReset(t, normalizer)
		if _, err := normalizer.push([]byte("data: x\n\n")); err == nil {
			t.Fatal("failed normalizer accepted more input")
		}
	})
}

func assertResponsesSSEScanCursorReset(t *testing.T, normalizer *responsesSSEFramingNormalizer) {
	t.Helper()
	if normalizer.scanOffset != 0 || normalizer.scanLineStart != 0 {
		t.Fatalf("cursor not reset: scanOffset=%d scanLineStart=%d", normalizer.scanOffset, normalizer.scanLineStart)
	}
	if normalizer.pending != nil {
		t.Fatalf("pending not cleared: %d bytes", len(normalizer.pending))
	}
}

// TestResponsesSSEFramingNormalizerChunkingEquivalence asserts that the full
// framing contract is invariant under chunk boundaries: whole input, fixed
// random, two/three byte and one-byte feeds must all produce byte-identical
// output and the same error category/message. It covers successful merges,
// every ambiguity rejection, partial EOF, UTF-8 fail-closed and the event-size
// limit boundary.
func TestResponsesSSEFramingNormalizerChunkingEquivalence(t *testing.T) {
	const eventName = "response.output_text.delta"
	validInput := "event: " + eventName + "\n\ndata: {\"type\":\"" + eventName + "\"}\n\n"
	limitExact := len(validInput) - 1
	limitOver := len(validInput) - 2

	cases := []struct {
		name  string
		limit int
		input []byte
	}{
		{
			name:  "merge lf",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n: keep-alive\n\ndata: {\"type\":\"" + eventName + "\",\"delta\":\"ok\"}\n\n"),
		},
		{
			name:  "merge crlf",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\r\n\r\ndata: {\"type\":\"" + eventName + "\"}\r\n\r\n"),
		},
		{
			name:  "merge cr",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\r\rdata: {\"type\":\"" + eventName + "\"}\r\r"),
		},
		{
			name:  "merge cr with trailing cr run",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\r\rdata: {\"type\":\"" + eventName + "\"}\r\r\r\r"),
		},
		{
			name:  "cr run blocks",
			limit: maxSSEEventBytes,
			input: bytes.Repeat([]byte(": keep-alive\r\r"), 64),
		},
		{
			name:  "consecutive empty blocks",
			limit: maxSSEEventBytes,
			input: bytes.Repeat([]byte("\n\n"), 64),
		},
		{
			name:  "crlf split across chunks",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\r\n: keep-alive\r\n\r\ndata: {\"type\":\"" + eventName + "\"}\r\n\r\n"),
		},
		{
			name:  "independent heartbeat",
			limit: maxSSEEventBytes,
			input: []byte(": keep-alive\n\ndata: {\"type\":\"response.completed\"}\n\n"),
		},
		{
			name:  "id only block",
			limit: maxSSEEventBytes,
			input: []byte("id: request-1\n\ndata: {\"type\":\"response.completed\"}\n\n"),
		},
		{
			name:  "mismatched type",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: {\"type\":\"response.completed\"}\n\n"),
		},
		{
			name:  "missing type",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: {\"delta\":\"x\"}\n\n"),
		},
		{
			name:  "duplicate type",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: {\"type\":\"" + eventName + "\",\"type\":\"" + eventName + "\"}\n\n"),
		},
		{
			name:  "non object payload",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: []\n\n"),
		},
		{
			name:  "empty data",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata:\n\n"),
		},
		{
			name:  "extra field in data block",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\nid: request-1\ndata: {\"type\":\"" + eventName + "\"}\n\n"),
		},
		{
			name:  "non adjacent block",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\n: keep-alive\n\ndata: {\"type\":\"" + eventName + "\"}\n\n"),
		},
		{
			name:  "done payload",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: [DONE]\n\n"),
		},
		{
			name:  "duplicate event",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\nevent: " + eventName + "\n\ndata: {\"type\":\"" + eventName + "\"}\n\n"),
		},
		{
			name:  "empty event",
			limit: maxSSEEventBytes,
			input: []byte("event:\n\ndata: {\"type\":\"" + eventName + "\"}\n\n"),
		},
		{
			name:  "invalid utf8 comment",
			limit: maxSSEEventBytes,
			input: append(append([]byte("event: "+eventName+"\n: "), 0xff), []byte("\n\ndata: {\"type\":\""+eventName+"\"}\n\n")...),
		},
		{
			name:  "invalid utf8 payload",
			limit: maxSSEEventBytes,
			input: append(append([]byte("event: "+eventName+"\n\ndata: {\"type\":\""+eventName+"\",\"delta\":\""), 0xff), []byte("\"}\n\n")...),
		},
		{
			name:  "partial eof event only",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\n"),
		},
		{
			name:  "partial eof data unterminated",
			limit: maxSSEEventBytes,
			input: []byte("event: " + eventName + "\n\ndata: {\"type\":\"" + eventName + "\"}\n"),
		},
		{
			name:  "partial eof trailing cr",
			limit: maxSSEEventBytes,
			input: []byte("data: {\"type\":\"response.completed\"}\r"),
		},
		{
			name:  "event limit exact",
			limit: limitExact,
			input: []byte(validInput),
		},
		{
			name:  "event limit over",
			limit: limitOver,
			input: []byte(validInput),
		},
		{
			name:  "oversized single line",
			limit: 64,
			input: bytes.Repeat([]byte("x"), 4096),
		},
		{
			name:  "many independent blocks",
			limit: maxSSEEventBytes,
			input: bytes.Repeat([]byte(": keep-alive\n\n"), 256),
		},
	}

	chunkings := []struct {
		name  string
		chunk func([]byte) [][]byte
	}{
		{name: "one byte", chunk: chunkedBySize(1)},
		{name: "two bytes", chunk: chunkedBySize(2)},
		{name: "three bytes", chunk: chunkedBySize(3)},
		{name: "split at cr", chunk: chunkedSplitAtCarriageReturn},
		{name: "fixed random", chunk: chunkedRandomly(1, 13, 424242)},
		{name: "whole input", chunk: chunkedBySize(1 << 20)},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			reference := runResponsesSSEFraming(test.input, test.limit, chunkedBySize(1<<20))
			for _, chunking := range chunkings {
				got := runResponsesSSEFraming(test.input, test.limit, chunking.chunk)
				if !bytes.Equal(got.output, reference.output) {
					t.Fatalf("chunking %q output = %q, want %q", chunking.name, got.output, reference.output)
				}
				if framingOutcome(got.err) != framingOutcome(reference.err) {
					t.Fatalf("chunking %q outcome = %q, want %q", chunking.name, framingOutcome(got.err), framingOutcome(reference.err))
				}
			}
		})
	}
}

type responsesSSEFramingRun struct {
	output []byte
	err    error
}

func runResponsesSSEFraming(input []byte, limit int, chunk func([]byte) [][]byte) responsesSSEFramingRun {
	normalizer := newResponsesSSEFramingNormalizer(limit)
	var output bytes.Buffer
	for _, part := range chunk(input) {
		got, err := normalizer.push(part)
		if err != nil {
			return responsesSSEFramingRun{err: err}
		}
		output.Write(got)
	}
	got, err := normalizer.finish()
	if err != nil {
		return responsesSSEFramingRun{err: err}
	}
	output.Write(got)
	return responsesSSEFramingRun{output: output.Bytes()}
}

func framingOutcome(err error) string {
	switch {
	case err == nil:
		return "ok"
	case errors.Is(err, errSSEEventTooLarge):
		return "too-large: " + err.Error()
	case errors.Is(err, errSSEEventIncomplete):
		return "incomplete: " + err.Error()
	case errors.Is(err, errResponsesSSEFraming):
		return "framing: " + err.Error()
	default:
		return "other: " + err.Error()
	}
}

func chunkedBySize(size int) func([]byte) [][]byte {
	return func(input []byte) [][]byte { return splitBytes(input, size) }
}

func chunkedRandomly(minSize, maxSize int, seed int64) func([]byte) [][]byte {
	return func(input []byte) [][]byte {
		random := rand.New(rand.NewSource(seed))
		chunks := make([][]byte, 0, 16)
		for len(input) > 0 {
			size := minSize + random.Intn(maxSize-minSize+1)
			if size > len(input) {
				size = len(input)
			}
			chunks = append(chunks, bytes.Clone(input[:size]))
			input = input[size:]
		}
		return chunks
	}
}

func chunkedSplitAtCarriageReturn(input []byte) [][]byte {
	chunks := make([][]byte, 0, 4)
	start := 0
	for index := 0; index < len(input); index++ {
		if input[index] != '\r' {
			continue
		}
		chunks = append(chunks, bytes.Clone(input[start:index+1]))
		start = index + 1
	}
	if start < len(input) {
		chunks = append(chunks, bytes.Clone(input[start:]))
	}
	return chunks
}

func normalizeResponsesSSEFramingForTest(input []byte, chunkSize int) ([]byte, error) {
	normalizer := newResponsesSSEFramingNormalizer(maxSSEEventBytes)
	var output bytes.Buffer
	for _, chunk := range splitBytes(input, chunkSize) {
		part, err := normalizer.push(chunk)
		if err != nil {
			return nil, err
		}
		output.Write(part)
	}
	part, err := normalizer.finish()
	if err != nil {
		return nil, err
	}
	output.Write(part)
	return output.Bytes(), nil
}

func chunkSizeName(size int) string {
	return "chunk-" + strconv.Itoa(size)
}

func lineEndingName(lineEnding string) string {
	switch lineEnding {
	case "\n":
		return "LF"
	case "\r\n":
		return "CRLF"
	default:
		return "CR"
	}
}
