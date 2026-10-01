// Command webui serves the embedded management UI exactly as the production
// binary does — same page routes, assets, and CSP — without the data/control planes. The full gpt-load binary only compiles on
// Linux (internal/platform/securefile is linux-only), so this harness exists
// for local CSP/browser verification on any platform:
//
//	go build -o webui-harness ./internal/webui/cmd/webui
//	GPT_LOAD_BINARY=./webui-harness pnpm --dir web playwright test --project go-csp
package main

import (
	"log"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/webui"
)

func main() {
	server, err := webui.NewServer()
	if err != nil {
		log.Fatalf("webui server: %v", err)
	}
	registry, err := httproute.NewRegistry(server.HTTPModule())
	if err != nil {
		log.Fatalf("route registry: %v", err)
	}
	// Mirror the production engine's behavioral knobs (internal/app/app.go):
	// release mode, no trusted proxies, and no trailing-slash redirect — a
	// default gin engine would 301 "/settings/" where production 404s.
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.RedirectTrailingSlash = false
	if err := engine.SetTrustedProxies(nil); err != nil {
		log.Fatalf("disable trusted proxies: %v", err)
	}
	if err := registry.Bind(engine); err != nil {
		log.Fatalf("bind routes: %v", err)
	}

	host := os.Getenv("HOST")
	if host == "" {
		host = "127.0.0.1"
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "3001"
	}
	log.Printf("webui harness listening on http://%s:%s", host, port)
	if err := http.ListenAndServe(host+":"+port, engine); err != nil {
		log.Fatalf("serve: %v", err)
	}
}
