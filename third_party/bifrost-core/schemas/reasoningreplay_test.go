package schemas

import (
	"encoding/json"
	"reflect"
	"testing"
)

// Check the public wire, not just a non-nil embedded Go struct: summary:null
// and a missing summary both fail Responses input validation.
func assertReplayableReasoning(t *testing.T, item ResponsesMessage, text string, summary []ResponsesReasoningSummary) {
	t.Helper()
	body, err := json.Marshal(item)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if _, exists := wire["role"]; exists {
		t.Errorf("reasoning item must not carry role: %s", body)
	}
	if raw := wire["summary"]; len(raw) == 0 || raw[0] != '[' {
		t.Errorf("reasoning summary must be an array: %s", body)
	} else {
		var got []ResponsesReasoningSummary
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, summary) {
			t.Errorf("summary = %#v, want %#v", got, summary)
		}
	}
	if text != "" && (item.Content == nil || len(item.Content.ContentBlocks) != 1 ||
		item.Content.ContentBlocks[0].Type != ResponsesOutputMessageContentTypeReasoning ||
		item.Content.ContentBlocks[0].Text == nil || *item.Content.ContentBlocks[0].Text != text) {
		t.Errorf("original reasoning text was lost or recast as a summary: %s", body)
	}
}

func TestChatReasoningReplayUnary(t *testing.T) {
	for _, test := range []struct {
		name    string
		message ChatAssistantMessage
		text    string
		summary []ResponsesReasoningSummary
	}{
		{name: "plain", message: ChatAssistantMessage{Reasoning: Ptr("plan")}, text: "plan", summary: []ResponsesReasoningSummary{}},
		{name: "text details", message: ChatAssistantMessage{ReasoningDetails: []ChatReasoningDetails{
			{Type: BifrostReasoningDetailsTypeText, Text: Ptr("plan"), Signature: Ptr("signature")},
		}}, text: "plan", summary: []ResponsesReasoningSummary{}},
		{name: "summary and encrypted", message: ChatAssistantMessage{ReasoningDetails: []ChatReasoningDetails{
			{Type: BifrostReasoningDetailsTypeSummary, Summary: Ptr("short summary")},
			{Type: BifrostReasoningDetailsTypeEncrypted, Data: Ptr("opaque")},
		}}, summary: []ResponsesReasoningSummary{{Type: ResponsesReasoningContentBlockTypeSummaryText, Text: "short summary"}}},
		{name: "encrypted only", message: ChatAssistantMessage{ReasoningDetails: []ChatReasoningDetails{
			{Type: BifrostReasoningDetailsTypeEncrypted, Data: Ptr("opaque")},
		}}, summary: []ResponsesReasoningSummary{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			message := ChatMessage{Role: ChatMessageRoleAssistant, Content: &ChatMessageContent{ContentStr: Ptr("answer")}, ChatAssistantMessage: &test.message}
			response := (&BifrostChatResponse{Choices: []BifrostResponseChoice{{
				FinishReason: Ptr("stop"), ChatNonStreamResponseChoice: &ChatNonStreamResponseChoice{Message: &message},
			}}}).ToBifrostResponsesResponse()
			if len(response.Output) != 2 || response.Output[0].Type == nil || *response.Output[0].Type != ResponsesMessageTypeReasoning {
				t.Fatalf("reasoning output = %#v", response.Output)
			}
			item := response.Output[0]
			assertReplayableReasoning(t, item, test.text, test.summary)
			for _, detail := range test.message.ReasoningDetails {
				if detail.Data != nil && (item.ResponsesReasoning == nil || item.EncryptedContent == nil || *item.EncryptedContent != *detail.Data) {
					t.Error("encrypted content was lost")
				}
				if detail.Signature != nil && (item.Content == nil || item.Content.ContentBlocks[0].Signature == nil || *item.Content.ContentBlocks[0].Signature != *detail.Signature) {
					t.Error("text signature was lost")
				}
			}
		})
	}
	message := ChatMessage{Role: ChatMessageRoleAssistant, Content: &ChatMessageContent{ContentStr: Ptr("answer")}}
	for _, item := range message.ToResponsesMessages() {
		if item.Type != nil && *item.Type == ResponsesMessageTypeReasoning {
			t.Fatal("a message without reasoning gained an empty reasoning item")
		}
	}
}

func TestChatReasoningReplayStream(t *testing.T) {
	for _, finish := range []string{"tool_calls", "length"} {
		t.Run(finish, func(t *testing.T) {
			state := AcquireChatToResponsesStreamState()
			defer ReleaseChatToResponsesStreamState(state)
			chunks := []*BifrostChatResponse{
				streamChunk("c1", &ChatStreamResponseChoiceDelta{Reasoning: Ptr("plan")}, nil),
				streamChunk("c1", &ChatStreamResponseChoiceDelta{ToolCalls: []ChatAssistantMessageToolCall{{
					ID: Ptr("call_1"), Type: Ptr("function"),
					Function: ChatAssistantMessageToolCallFunction{Name: Ptr("lookup"), Arguments: `{}`},
				}}}, nil),
				streamChunk("c1", &ChatStreamResponseChoiceDelta{}, &finish),
			}
			var done *ResponsesMessage
			var terminal *BifrostResponsesResponse
			added := false
			for _, chunk := range chunks {
				for _, event := range chunk.ToBifrostResponsesStreamResponse(state) {
					if event.Item != nil && event.Item.Type != nil && *event.Item.Type == ResponsesMessageTypeReasoning {
						text := "plan"
						if event.Type == ResponsesStreamResponseTypeOutputItemAdded {
							text, added = "", true
						}
						assertReplayableReasoning(t, *event.Item, text, []ResponsesReasoningSummary{})
						if event.Type == ResponsesStreamResponseTypeOutputItemDone {
							done = event.Item
						}
					}
					if event.Response != nil && (event.Type == ResponsesStreamResponseTypeCompleted || event.Type == ResponsesStreamResponseTypeIncomplete) {
						terminal = event.Response
					}
				}
			}
			if !added || done == nil || terminal == nil || len(terminal.Output) < 2 {
				t.Fatalf("incomplete conversion: added=%v done=%v terminal=%#v", added, done, terminal)
			}
			assertReplayableReasoning(t, terminal.Output[0], "plan", []ResponsesReasoningSummary{})
			if !reflect.DeepEqual(*done, terminal.Output[0]) {
				t.Error("done reasoning differs from terminal output")
			}
			// Replay the actual wire objects without repairing them in the test.
			body, err := json.Marshal(terminal.Output)
			if err != nil {
				t.Fatal(err)
			}
			var replay []ResponsesMessage
			if err := json.Unmarshal(body, &replay); err != nil {
				t.Fatal(err)
			}
			replay = append(replay, ResponsesMessage{Type: Ptr(ResponsesMessageTypeFunctionCallOutput), ResponsesToolMessage: &ResponsesToolMessage{
				CallID: Ptr("call_1"), Output: &ResponsesToolMessageOutputStruct{ResponsesToolCallOutputStr: Ptr("result")},
			}})
			chat := (&BifrostResponsesRequest{Input: replay}).ToChatRequest()
			if len(chat.Input) != 2 || chat.Input[0].ChatAssistantMessage == nil || chat.Input[0].Reasoning == nil || *chat.Input[0].Reasoning != "plan" ||
				len(chat.Input[0].ToolCalls) != 1 || chat.Input[0].ToolCalls[0].ID == nil || *chat.Input[0].ToolCalls[0].ID != "call_1" ||
				chat.Input[1].ChatToolMessage == nil || chat.Input[1].ToolCallID == nil || *chat.Input[1].ToolCallID != "call_1" {
				t.Fatalf("reasoning/tool pairing lost on replay: %#v", chat.Input)
			}
		})
	}
}
