package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jacek/fluxgate/internal/balancer"
	"github.com/jacek/fluxgate/internal/gateway"
	"github.com/jacek/fluxgate/internal/limiter"
	"github.com/jacek/fluxgate/internal/modelrouter"
	"github.com/jacek/fluxgate/internal/provider"
	"github.com/jacek/fluxgate/internal/proxy"
	pkgconfig "github.com/jacek/fluxgate/pkg/config"
)

type Server struct {
	cfg          *pkgconfig.Config
	server       *http.Server
	gateway      *gateway.Gateway
	registry     *provider.Registry
	modelRouter  *modelrouter.Router
	proxyRegistry *proxy.Registry
	logger       *slog.Logger
	healthCtx    context.Context
	healthCancel context.CancelFunc
}

func New(cfg *pkgconfig.Config) (*Server, error) {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// Build proxy registry
	var proxyConfigs []proxy.Config
	for _, pc := range cfg.Proxies {
		proxyConfigs = append(proxyConfigs, proxy.Config{
			Name:            pc.Name,
			Type:            pc.Type,
			Address:         pc.Address,
			Username:        pc.Username,
			Password:        pc.Password,
			MaxIdleConns:    pc.MaxIdleConns,
			MaxConnsPerHost: pc.MaxConnsPerHost,
			IdleConnTimeout: pc.IdleConnTimeout,
		})
	}
	cbConfig := proxy.CircuitBreakerConfig{
		FailureThreshold: 5,
		RecoveryTimeout:  30 * time.Second,
		HalfOpenMax:     3,
	}
	proxyRegistry := proxy.NewRegistry(proxyConfigs, cbConfig)

	// Build provider registry
	registry := provider.NewRegistry()
	for _, pc := range cfg.Providers {
		providerType := pc.Type
		if providerType == "" {
			providerType = provider.DetectProviderType(pc.BaseURL)
		}
		p := provider.New(pc.Name, providerType, pc.BaseURL, pc.APIKeys, pc.Weight)
		p.ProxyName = pc.Proxy
		p.Models = pc.Models
		p.Disabled = pc.Disabled
		registry.Add(p)
	}

	// Build model router
	modelRouter, err := modelrouter.NewRouter(cfg.Models, registry)
	if err != nil {
		return nil, fmt.Errorf("building model router: %w", err)
	}

	// Auto-compute rate limit: each key gets ~5 RPS, burst = keys × 10
	totalKeys := cfg.TotalKeyCount()
	rps := float64(totalKeys) * 5
	burst := totalKeys * 10
	if rps < 100 {
		rps = 100
	}
	if burst < 200 {
		burst = 200
	}

	bal := balancer.NewBalancer(cfg.Gateway.BalancerType)
	lim := limiter.NewRateLimiter(rps, burst)

	gw := gateway.New(registry, bal, lim, cfg.Gateway, proxyRegistry, modelRouter)

	s := &Server{
		cfg:           cfg,
		gateway:       gw,
		registry:      registry,
		modelRouter:   modelRouter,
		proxyRegistry: proxyRegistry,
		logger:        logger,
	}

	mux := http.NewServeMux()

	// Health endpoint: no auth required
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `{"status":"ok"}`)
	})

	// Auth-protected routes
	mux.Handle("/v1/models", authMiddleware(cfg.Gateway.APIKeys, http.HandlerFunc(s.handleModels)))
	mux.Handle("/", authMiddleware(cfg.Gateway.APIKeys, gw))

	addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
	s.server = &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
	}

	return s, nil
}

func (s *Server) Start() error {
	errCh := make(chan error, 1)

	go func() {
		s.logger.Info("starting server", "addr", s.server.Addr)
		if err := s.server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	// Health checker
	if s.cfg.Gateway.HealthCheck {
		s.healthCtx, s.healthCancel = context.WithCancel(context.Background())
		go s.runHealthChecker()
	}

	// Wait for interrupt
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	select {
	case err := <-errCh:
		return fmt.Errorf("server error: %w", err)
	case sig := <-quit:
		s.logger.Info("shutting down", "signal", sig)
	}

	// Stop health checker
	if s.healthCancel != nil {
		s.healthCancel()
	}

	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := s.server.Shutdown(ctx); err != nil {
		return fmt.Errorf("server shutdown error: %w", err)
	}

	// Close proxy connections
	if s.proxyRegistry != nil {
		s.proxyRegistry.Close()
	}

	s.logger.Info("server stopped")
	return nil
}

func (s *Server) runHealthChecker() {
	ticker := time.NewTicker(s.cfg.Gateway.HealthInterval)
	defer ticker.Stop()

	for {
		select {
		case <-s.healthCtx.Done():
			return
		case <-ticker.C:
			ctx, cancel := context.WithTimeout(context.Background(), s.cfg.Gateway.HealthTimeout)
			s.gateway.HealthCheck(ctx)
			cancel()
		}
	}
}

func (s *Server) ListenAddr() string {
	return s.server.Addr
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		OwnedBy string `json:"owned_by"`
	}

	seen := make(map[string]bool)
	var models []model

	// Virtual models from the model router
	if s.modelRouter != nil {
		for _, entry := range s.modelRouter.AllEntries() {
			if !seen[entry.VirtualModel] {
				seen[entry.VirtualModel] = true
				models = append(models, model{
					ID:      entry.VirtualModel,
					Object:  "model",
					Created: 0,
					OwnedBy: "fluxgate",
				})
			}
		}
	}

	// Real models from providers
	for _, p := range s.registry.All() {
		for _, m := range p.Models {
			if !seen[m] {
				seen[m] = true
				models = append(models, model{
					ID:      m,
					Object:  "model",
					Created: 0,
					OwnedBy: p.Name,
				})
			}
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"object": "list",
		"data":   models,
	})
}
