package i18n

import (
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestMiddlewareSkipsJapanese(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	gin.SetMode(gin.TestMode)
	for _, test := range []struct {
		header, language, message string
	}{
		{header: "ja-JP,ja;q=0.9", language: "zh-CN", message: "请求错误"},
		{header: "ja-JP, en-US;q=0.9", language: "en-US", message: "Bad request"},
	} {
		t.Run(test.header, func(t *testing.T) {
			context, _ := gin.CreateTestContext(nil)
			context.Request = httptest.NewRequest("GET", "/", nil)
			context.Request.Header.Set("Accept-Language", test.header)
			Middleware()(context)

			if got := GetLanguageFromContext(context); got != test.language {
				t.Errorf("GetLanguageFromContext() = %q, want %q", got, test.language)
			}
			if got := Message(context, "bad_request"); got != test.message {
				t.Errorf("Message() = %q, want %q", got, test.message)
			}
		})
	}
}

func TestRequestTooLargeTranslations(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	for _, test := range []struct {
		language string
		message  string
	}{
		{language: "zh-CN", message: "请求体过大"},
		{language: "en-US", message: "Request body is too large"},
		{language: "ja-JP", message: "请求体过大"},
	} {
		t.Run(test.language, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			context, _ := gin.CreateTestContext(nil)
			context.Request = httptest.NewRequest("GET", "/", nil)
			context.Request.Header.Set("Accept-Language", test.language)
			Middleware()(context)

			if got := Message(context, "request_too_large"); got != test.message {
				t.Fatalf("Message() = %q, want %q", got, test.message)
			}
		})
	}
}

func TestChannelTargetConflictTranslations(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	for _, test := range []struct {
		language string
		message  string
	}{
		{language: "zh-CN", message: "已有分组使用相同渠道目标"},
		{language: "en-US", message: "An existing group already uses this channel target"},
		{language: "ja-JP", message: "已有分组使用相同渠道目标"},
	} {
		t.Run(test.language, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			context, _ := gin.CreateTestContext(nil)
			context.Request = httptest.NewRequest("GET", "/", nil)
			context.Request.Header.Set("Accept-Language", test.language)
			Middleware()(context)

			if got := Message(context, "group.channel_target_conflict"); got != test.message {
				t.Fatalf("Message() = %q, want %q", got, test.message)
			}
		})
	}
}

func TestGroupInUseTranslations(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	for _, test := range []struct {
		language string
		message  string
	}{
		{language: "zh-CN", message: "分组仍被访问密钥引用"},
		{language: "en-US", message: "The group is still referenced by access keys"},
		{language: "ja-JP", message: "分组仍被访问密钥引用"},
	} {
		t.Run(test.language, func(t *testing.T) {
			gin.SetMode(gin.TestMode)
			context, _ := gin.CreateTestContext(nil)
			context.Request = httptest.NewRequest("GET", "/", nil)
			context.Request.Header.Set("Accept-Language", test.language)
			Middleware()(context)

			if got := Message(context, "group.in_use"); got != test.message {
				t.Fatalf("Message() = %q, want %q", got, test.message)
			}
		})
	}
}

func TestUnsupportedLanguageFallsBackToChinese(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(nil)
	context.Request = httptest.NewRequest("GET", "/", nil)
	context.Request.Header.Set("Accept-Language", "fr-FR")
	Middleware()(context)

	if got := Message(context, "bad_request"); got != "请求错误" {
		t.Fatalf("Message() = %q, want %q", got, "请求错误")
	}
}

func TestMiddlewareUsesSupportedAcceptLanguageFallback(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(nil)
	context.Request = httptest.NewRequest("GET", "/", nil)
	context.Request.Header.Set("Accept-Language", "fr-FR, en-US;q=0.9")
	Middleware()(context)

	if got := Message(context, "bad_request"); got != "Bad request" {
		t.Fatalf("Message() = %q, want %q", got, "Bad request")
	}
}

func TestAcceptLanguageIgnoresUnrecognizedRanges(t *testing.T) {
	if got := ResolveLanguage("en-US, ac;q=0.9"); got != "en-US" {
		t.Fatalf("ResolveLanguage() = %q, want %q", got, "en-US")
	}
}

func TestAcceptLanguageWildcardUsesServerDefault(t *testing.T) {
	if got := ResolveLanguage("*;q=1, en-US;q=0.8"); got != "zh-CN" {
		t.Fatalf("ResolveLanguage() = %q, want %q", got, "zh-CN")
	}
}

func TestAcceptLanguagePreservesQualityOrder(t *testing.T) {
	if got := ResolveLanguage("en-US;q=0.8, zh-CN;q=0.9"); got != "zh-CN" {
		t.Fatalf("ResolveLanguage() = %q, want %q", got, "zh-CN")
	}
}

func TestAcceptLanguageOversizedHeaderUsesServerDefault(t *testing.T) {
	header := strings.Repeat("a", maxAcceptLanguageBytes+1) + ", en-US"
	if got := ResolveLanguage(header); got != "zh-CN" {
		t.Fatalf("ResolveLanguage() = %q, want %q", got, "zh-CN")
	}
}

func TestAcceptLanguageWithTooManyEntriesUsesServerDefault(t *testing.T) {
	header := strings.Repeat("fr-FR,", maxAcceptLanguageEntries) + "en-US"
	if got := ResolveLanguage(header); got != "zh-CN" {
		t.Fatalf("ResolveLanguage() = %q, want %q", got, "zh-CN")
	}
}

func TestMessageWithoutLocalizerUsesChinese(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init() error = %v", err)
	}

	gin.SetMode(gin.TestMode)
	context, _ := gin.CreateTestContext(nil)

	if got := Message(context, "bad_request"); got != "请求错误" {
		t.Fatalf("Message() = %q, want %q", got, "请求错误")
	}
}
