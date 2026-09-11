package control

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"testing"
)

// TestFrontendHealthAllowlistCoversWireKeys locks the cross-end invariant:
// the set of top-level JSON keys of backend runtimeHealthResponse (served at
// /api/health) must be a subset of the frontend healthFields allowlist in
// web/src/app/resources/health.ts. This catches the drift class "backend adds a
// field, frontend allowlist not updated" at `go test ./internal/control/...`.
//
// Design (per U003 scope):
//  1. backend keys come from reflecting runtimeHealthResponse json tags (no hand list)
//  2. frontend path default ../../web/src/app/resources/health.ts relative to this
//     test source dir; overridable via FRONTEND_HEALTH_TS; CWD-independent
//  3. parse healthFields array literal, extract string literals
//  4. parser self-check after parse: >=13 keys and must contain observed_at_ms,
//     request_log, debug_capture; otherwise t.Fatal (never t.Skip, never silent),
//     message opens with "PARSER-SELFCHECK-FAIL" to distinguish from "allowlist missing"
//  5. assertion: backend keys subset of frontend allowlist; list all missing at once
//  6. extra frontend keys are t.Logf advisory only
//
// No node / pnpm / frontend toolchain dependency.
func TestFrontendHealthAllowlistCoversWireKeys(t *testing.T) {
	backendKeys := backendHealthWireKeys(t)
	healthTSPath := resolveFrontendHealthTS(t)

	source, err := os.ReadFile(healthTSPath)
	if err != nil {
		t.Fatalf("PARSER-SELFCHECK-FAIL: cannot read frontend health.ts (%s): %v", healthTSPath, err)
	}
	frontendKeys, parseErr := extractHealthFields(string(source))
	if parseErr != nil {
		t.Fatalf("PARSER-SELFCHECK-FAIL: %s (%s)", parseErr, healthTSPath)
	}

	// 解析器自检（白名单守卫）：若 health.ts 被截断或解析器失效，
	// 必须一眼看出是「解析器失效」，而非「白名单缺字段」。
	if len(frontendKeys) < 13 {
		t.Fatalf(
			"解析器失效: healthFields 仅解析到 %d 个字段，期望 >= 13（疑似 health.ts 被截断或解析器失效）",
			len(frontendKeys),
		)
	}
	for _, anchor := range []string{"observed_at_ms", "request_log", "debug_capture"} {
		if !containsString(frontendKeys, anchor) {
			t.Fatalf(
				"解析器失效: healthFields 未包含必含字段 %s（解析器守卫：白名单应恒定包含该后端键；缺失可能意味着 health.ts 被改或解析器失效）",
				anchor,
			)
		}
	}

	// 断言：后端键集合 ⊆ 前端白名单；缺失键一次性全部列出。
	frontendSet := make(map[string]struct{}, len(frontendKeys))
	for _, k := range frontendKeys {
		frontendSet[k] = struct{}{}
	}
	var missing []string
	for _, k := range backendKeys {
		if _, ok := frontendSet[k]; !ok {
			missing = append(missing, k)
		}
	}
	sort.Strings(missing)
	if len(missing) > 0 {
		t.Fatalf(
			"白名单缺字段: 后端 runtimeHealthResponse 含以下顶层键但前端 healthFields 白名单缺失 -> %s",
			strings.Join(missing, ", "),
		)
	}

	// advisory：前端多出的键（后端未发射，无害），仅记录不失败。
	backendSet := make(map[string]struct{}, len(backendKeys))
	for _, k := range backendKeys {
		backendSet[k] = struct{}{}
	}
	for _, k := range frontendKeys {
		if _, ok := backendSet[k]; !ok {
			t.Logf("advisory: 前端白名单多出键 %s（后端未发射，无害）", k)
		}
	}
}

// backendHealthWireKeys 通过反射 runtimeHealthResponse 取得权威后端顶层 JSON 键集合，
// 忽略 "-" 与带选项的 tag（按逗号切分取首段）。不手抄任何键列表。
func backendHealthWireKeys(t *testing.T) []string {
	t.Helper()
	rt := reflect.TypeOf(runtimeHealthResponse{})
	if rt == nil {
		t.Fatal("解析器失效: 无法反射 runtimeHealthResponse 类型")
	}
	var keys []string
	for i := 0; i < rt.NumField(); i++ {
		field := rt.Field(i)
		tag := field.Tag.Get("json")
		if tag == "" {
			t.Fatalf("解析器失效: 字段 %s 缺少 json tag", field.Name)
		}
		name := strings.Split(tag, ",")[0]
		if name == "" || name == "-" {
			continue
		}
		keys = append(keys, name)
	}
	if len(keys) == 0 {
		t.Fatal("解析器失效: 未从 runtimeHealthResponse 反射到任何 json 键")
	}
	return keys
}

// resolveFrontendHealthTS 解析前端 health.ts 路径：
// 默认相对本测试源文件目录（internal/control）为 ../../web/src/app/resources/health.ts，
// 可用环境变量 FRONTEND_HEALTH_TS 覆盖。基于 runtime.Caller，与运行目录无关。
func resolveFrontendHealthTS(t *testing.T) string {
	t.Helper()
	if override := os.Getenv("FRONTEND_HEALTH_TS"); override != "" {
		return override
	}
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("解析器失效: 无法通过 runtime.Caller 定位测试源文件以解析前端 health.ts 路径")
	}
	pkgDir := filepath.Dir(thisFile)
	return filepath.Join(pkgDir, "..", "..", "web", "src", "app", "resources", "health.ts")
}

// extractHealthFields 定位 `const healthFields = [ ... ] as const` 并提取其中的字符串字面量。
// 任何解析失败返回非 nil error（调用方转为「解析器失效」Fatal）。
func extractHealthFields(source string) ([]string, error) {
	const decl = "const healthFields"
	idx := strings.Index(source, decl)
	if idx < 0 {
		return nil, fmt.Errorf("未定位 %q 声明", decl)
	}
	rest := source[idx+len(decl):]
	open := strings.Index(rest, "[")
	if open < 0 {
		return nil, fmt.Errorf("healthFields 数组字面量缺少起始 '['")
	}
	close := strings.Index(rest[open:], "]")
	if close < 0 {
		return nil, fmt.Errorf("healthFields 数组字面量未闭合 ']'（疑似被截断）")
	}
	arrayBody := rest[open+1 : open+close]

	literalRE := regexp.MustCompile("'([^']*)'|\"([^\"]*)\"")
	var keys []string
	for _, m := range literalRE.FindAllStringSubmatch(arrayBody, -1) {
		if m[1] != "" {
			keys = append(keys, m[1])
		} else if m[2] != "" {
			keys = append(keys, m[2])
		}
	}
	return keys, nil
}

func containsString(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
