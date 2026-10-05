package channel

import (
	"gpt-load/internal/execution"
	"gpt-load/internal/protocol"
	"testing"
)

func TestU006PresetCapabilities(t *testing.T) {
	r := NewRegistry()
	for _, id := range []ID{"cerebras", "mistral", "nebius", "parasail", "wafer", "huggingface", "opencode_go", "opencode_zen"} {
		t.Run(string(id), func(t *testing.T) {
			d, ok := r.Get(id)
			if !ok {
				t.Fatalf("missing preset %s", id)
			}
			if d.Connection.CredentialInput != "batch_text" {
				t.Fatalf("credential input = %s", d.Connection.CredentialInput)
			}
			target, err := r.Resolve(id, nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := target.ProbeContract(); !ok {
				t.Fatal("missing local probe contract")
			}
			def, _ := r.lookup(id)
			openCode := id == "opencode_go" || id == "opencode_zen"
			for _, p := range []protocol.Protocol{protocol.OpenAICompletions, protocol.OpenAIResponses, protocol.Anthropic, protocol.Gemini} {
				op := execution.OperationChatCompletion
				if p == protocol.OpenAIResponses {
					op = execution.OperationResponsesCreate
				}
				mode, present := def.modes[p][op]
				if openCode && p == protocol.Gemini {
					if present {
						t.Fatal("OpenCode advertises Gemini")
					}
					continue
				}
				want := RouteConverted
				if p == protocol.OpenAICompletions || openCode {
					want = RouteNative
				}
				if !present || mode != want {
					t.Fatalf("%s mode=%s present=%t want=%s", p, mode, present, want)
				}
			}
			_, embeds := def.modes[protocol.OpenAIEmbeddings][execution.OperationEmbeddingsCreate]
			if embeds != (id == "mistral" || id == "nebius") {
				t.Fatalf("embeddings = %t", embeds)
			}
			for _, p := range []protocol.Protocol{protocol.OpenAIImages, protocol.Rerank} {
				if len(def.modes[p]) != 0 {
					t.Fatalf("unsupported %s", p)
				}
			}
		})
	}
}
