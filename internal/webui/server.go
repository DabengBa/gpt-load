// Package webui serves the embedded management UI on explicit page routes.
package webui

import (
	"embed"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

const (
	distRoot = "dist"
	indexCSP = "default-src 'self'; script-src 'self'; style-src 'self'; " +
		"style-src-elem 'self'; style-src-attr 'unsafe-inline'; " +
		"img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; " +
		"base-uri 'self'; frame-ancestors 'none'; form-action 'self'"
	fallbackIndex = `<!doctype html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPT-Load</title></head>
<body><main><h1>GPT-Load</h1><p>前端资源尚未构建，请运行 make build。</p></main></body>
</html>`
)

//go:embed all:dist
var embeddedFiles embed.FS

// Server serves immutable assets and the SPA index for known UI routes.
type Server struct {
	files       fs.FS
	root        string
	index       []byte
	astryxIndex []byte
	pages       []pageRoute
}

// NewServer creates an embedded UI server.
func NewServer() (*Server, error) {
	pages, err := loadPageRoutes()
	if err != nil {
		return nil, err
	}
	return newServerWithPages(embeddedFiles, distRoot, pages), nil
}

func newServer(files fs.FS, root string) *Server {
	pages, err := loadPageRoutes()
	if err != nil {
		panic(err)
	}
	return newServerWithPages(files, root, pages)
}

func newServerWithPages(files fs.FS, root string, pages []pageRoute) *Server {
	index, err := fs.ReadFile(files, path.Join(root, "index.html"))
	if err != nil {
		index = []byte(fallbackIndex)
	}

	// astryxIndex stays nil when the entry is absent: builds without the React
	// frontend behave exactly as before.
	astryxIndex, err := fs.ReadFile(files, path.Join(root, "astryx.html"))
	if err != nil {
		astryxIndex = nil
	}

	return &Server{
		files:       files,
		root:        root,
		index:       index,
		astryxIndex: astryxIndex,
		pages:       append([]pageRoute(nil), pages...),
	}
}

func acceptsHTML(value string) bool {
	for _, candidate := range strings.Split(value, ",") {
		mediaType, parameters, err := mime.ParseMediaType(strings.TrimSpace(candidate))
		if err != nil || (mediaType != "text/html" && mediaType != "application/xhtml+xml") {
			continue
		}
		quality := 1.0
		if rawQuality, ok := parameters["q"]; ok {
			quality, err = strconv.ParseFloat(rawQuality, 64)
		}
		if err == nil && quality > 0 && quality <= 1 {
			return true
		}
	}
	return false
}

// frontendCookieName selects the frontend document; it only ever picks
// between two embedded static files and must never influence auth, API
// routing, or file paths.
const frontendCookieName = "gpt-load.frontend"

func frontendPreference(c *gin.Context) string {
	value, err := c.Cookie(frontendCookieName)
	if err != nil {
		return "classic"
	}
	return value
}

func (s *Server) indexFor(c *gin.Context, page pageRoute) []byte {
	if s.astryxIndex != nil && page.Astryx && frontendPreference(c) == "astryx" {
		return s.astryxIndex
	}
	return s.index
}

func (s *Server) indexForNotFound(c *gin.Context) []byte {
	if s.astryxIndex != nil && frontendPreference(c) == "astryx" {
		return s.astryxIndex
	}
	return s.index
}

func (s *Server) servePage(page pageRoute) gin.HandlerFunc {
	return func(c *gin.Context) {
		s.serveIndexWithStatus(c, http.StatusOK, s.indexFor(c, page))
	}
}

func (s *Server) serveNotFoundIndex(c *gin.Context) {
	s.serveIndexWithStatus(c, http.StatusNotFound, s.indexForNotFound(c))
}

func (s *Server) serveIndexWithStatus(c *gin.Context, status int, document []byte) {
	c.Header("Cache-Control", "no-cache")
	c.Header("Vary", "Cookie")
	c.Header("Content-Security-Policy", indexCSP)
	c.Header("X-Content-Type-Options", "nosniff")
	c.Header("X-Frame-Options", "DENY")
	c.Data(status, "text/html; charset=utf-8", document)
}

func (s *Server) serveThemeBootstrap(c *gin.Context) {
	content, err := fs.ReadFile(s.files, path.Join(s.root, "theme-bootstrap.js"))
	if err != nil {
		c.Status(http.StatusNotFound)
		return
	}

	c.Header("Cache-Control", "no-cache")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Data(http.StatusOK, "text/javascript; charset=utf-8", content)
}

func (s *Server) serveFavicon(c *gin.Context) {
	content, err := fs.ReadFile(s.files, path.Join(s.root, "favicon.svg"))
	if err != nil {
		c.Status(http.StatusNotFound)
		return
	}

	c.Header("Cache-Control", "no-cache")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Data(http.StatusOK, "image/svg+xml", content)
}

func (s *Server) serveAsset(c *gin.Context) {
	assetPath := path.Clean(strings.TrimPrefix(c.Param("filepath"), "/"))
	if assetPath == "." || assetPath == "" || strings.HasPrefix(assetPath, "../") {
		c.Status(http.StatusNotFound)
		return
	}

	content, err := fs.ReadFile(s.files, path.Join(s.root, "assets", assetPath))
	if err != nil {
		c.Status(http.StatusNotFound)
		return
	}

	contentType := mime.TypeByExtension(path.Ext(assetPath))
	if contentType == "" {
		contentType = "application/octet-stream"
	}
	c.Header("Cache-Control", "public, max-age=31536000, immutable")
	c.Header("X-Content-Type-Options", "nosniff")
	c.Data(http.StatusOK, contentType, content)
}
