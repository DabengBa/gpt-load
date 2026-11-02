package control

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"os"
	"time"

	"github.com/sirupsen/logrus"

	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/releasecheck"
)

// ReleaseUpdateChecker is the control-plane on-demand view of the public release checker.
type ReleaseUpdateChecker interface {
	Check(context.Context, bool) (*releasecheck.Update, error)
}

type Server struct {
	authDigest        [sha256.Size]byte
	service           *Service
	systemInfo        systemInfoResponse
	authFailures      *authFailureLimiter
	compareDigest     func([]byte, []byte) int
	logger            *logrus.Logger
	authFailureEvents *utils.RateLimitedEventCounter
	startedAt         time.Time
	now               func() time.Time
	releaseChecker    ReleaseUpdateChecker
}

func NewServer(cfg *config.Config, service *Service) *Server {
	now := time.Now
	if err := loadModelTraceConfigFromEnvironment(os.Getenv); err != nil {
		logrus.WithError(err).Error("invalid ModelTrace configuration; fingerprint detection is disabled")
	}

	return &Server{
		authDigest:    sha256.Sum256([]byte(cfg.AuthKey)),
		service:       service,
		systemInfo:    newSystemInfoResponse(cfg),
		authFailures:  newAuthFailureLimiter(),
		compareDigest: subtle.ConstantTimeCompare,
		logger:        logrus.StandardLogger(),
		startedAt:     now().UTC(),
		now:           now,
		authFailureEvents: utils.NewRateLimitedEventCounter(
			time.Minute,
			time.Now,
		),
	}
}

// NewServerWithReleaseUpdateChecker wires the on-demand public update checker.
func NewServerWithReleaseUpdateChecker(
	cfg *config.Config,
	service *Service,
	releaseChecker ReleaseUpdateChecker,
) *Server {
	server := NewServer(cfg, service)
	server.releaseChecker = releaseChecker
	return server
}
