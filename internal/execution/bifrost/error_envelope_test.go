package bifrost

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

// 部署里的第三方聚合站直接返回顶层 {"code","message"}，不套 error 对象。
// 只认 error.* 会让「密钥被拒」这类证据退化成不可分类的客户端错误，
// 网关因此既不换候选也不计入凭据失败。
func TestPassthroughHTTPErrorReadsTopLevelProviderEnvelope(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		wantHint execution.FailureHint
		wantCode string
	}{
		{
			name:     "top level invalid key",
			status:   http.StatusForbidden,
			body:     `{"code":"INVALID_API_KEY","message":"your key was rejected"}`,
			wantHint: execution.FailureHintInvalidCredential,
			wantCode: "INVALID_API_KEY",
		},
		{
			name:     "top level model rejection",
			status:   http.StatusForbidden,
			body:     `{"code":"unsupported_model","message":"this key cannot use the model"}`,
			wantHint: execution.FailureHintModelUnavailable,
			wantCode: "unsupported_model",
		},
		{
			name:   "top level without markers stays unscoped",
			status: http.StatusForbidden,
			body:   `{"code":"permission_denied","message":"blocked by policy"}`,

			wantCode: "permission_denied",
		},
		{
			name:     "nested envelope keeps precedence",
			status:   http.StatusForbidden,
			body:     `{"code":"permission_denied","error":{"code":"invalid_api_key","message":"key rejected"}}`,
			wantHint: execution.FailureHintInvalidCredential,
			wantCode: "invalid_api_key",
		},
		{
			name:     "top level falls back when the nested object is empty",
			status:   http.StatusForbidden,
			body:     `{"code":"INVALID_API_KEY","error":{}}`,
			wantHint: execution.FailureHintInvalidCredential,
			wantCode: "INVALID_API_KEY",
		},
		{
			name:     "non json body stays unscoped",
			status:   http.StatusForbidden,
			body:     `<html><body>blocked</body></html>`,
			wantCode: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evidence := passthroughHTTPError(test.status, nil, []byte(test.body), nil)
			if evidence.Hint != test.wantHint || evidence.Code != test.wantCode {
				t.Fatalf(
					"evidence = %#v, want hint %q code %q",
					evidence, test.wantHint, test.wantCode,
				)
			}
		})
	}
}
